# §06 Hot Path Wiring — Q8_0 dispatch for Qwen2-class CPU MLP layers.
#
# Replaces MergedColumnParallelLinear (`.mlp.gate_up_proj`) and
# RowParallelLinear (`.mlp.down_proj`) quant_method with a dispatcher that
# calls torch.ops._C_cpu_ops.q8_0_linear. Weight is quantized to Q8_0
# (llama.cpp layout: 32-element blocks of int8 + fp16 scale) at load time;
# there is no runtime repack.
#
# Semantics (Phase A, 2026-04-19):
#   - Only layers that pass the strict filter below are patched. Patched
#     layers bypass IPEX / oneDNN at apply-time (their quant_method is
#     replaced by _Q8_0LinearMethod). All other layers (attention, head,
#     embedding, lm_head, MoE experts, vision / speech / audio towers)
#     keep whatever IPEX previously installed.
#   - Activations stay in BF16/FP32 on the Python side; the C++ kernel
#     does per-row dynamic INT8 quantize → VNNI dot → FP32 accumulate →
#     dtype-matched output.
#   - LoRA adapters are incompatible: static Q8_0 quantization of the
#     base weight freezes it, so runtime delta-W cannot be folded in.
#     When a LoRA config is present we skip the patch entirely.
#
# Scope filter (must match ALL):
#   - architecture ∈ _ALLOWED_ARCHS (currently {"Qwen2ForCausalLM"}).
#     Other families (Qwen2Moe, Qwen2_5_VL, Qwen2Audio, Qwen3, LLaMA…)
#     are not validated for this path and are skipped.
#   - module name ends with one of _ALLOWED_SUFFIXES
#   - module name does not contain any of _EXCLUDE_SUBSTRINGS
#
# Gating: HYBRID_VNNI_HOT_PATH=1 + _C_cpu_ops built + q8_0 ops registered
# + arch allowed + LoRA disabled. Any guard missing → silent no-op with
# an explanatory warning.

from __future__ import annotations

import os
import time
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

Q8_0_BLOCK_SIZE = 32
Q8_0_BLOCK_BYTES = 34  # 2 (fp16 scale) + 32 (int8 quants)

# Strict whitelist — only families whose MLP structure has been validated
# against this patch path. Multi-modal variants and MoE are intentionally
# excluded even though they may share module names.
_ALLOWED_ARCHS = frozenset({
    "Qwen2ForCausalLM",
})

_ALLOWED_SUFFIXES = (".mlp.gate_up_proj", ".mlp.down_proj")

# Substrings that disqualify a module even when the suffix matches.
# Covers MoE expert MLPs and multi-modal towers that re-use these names.
_EXCLUDE_SUBSTRINGS = (
    "experts.",
    "vision",
    "visual",
    "speech",
    "audio",
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _trace_enabled() -> bool:
    return _env_flag("VLLM_HYBRID_KERNEL_TRACE")


def _compute_qweight_nbytes(N: int, K: int) -> int:
    if K % Q8_0_BLOCK_SIZE != 0:
        raise ValueError(
            f"Q8_0 requires K % {Q8_0_BLOCK_SIZE} == 0, got K={K}")
    return N * (K // Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES


def _quantize_to_q8_0(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")
    if weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"weight dtype must be FP32 or BF16, got {weight.dtype}")
    weight_c = weight.contiguous().cpu()
    N, K = weight_c.shape
    nbytes = _compute_qweight_nbytes(N, K)
    qweight = torch.empty(nbytes, dtype=torch.int8)
    torch.ops._C_cpu_ops.q8_0_quantize_weight(qweight, weight_c)
    return qweight


class _Q8_0LinearMethod:
    """Wraps the original quant_method; overrides apply() to route through
    torch.ops._C_cpu_ops.q8_0_linear when layer has a quantized qweight buffer.

    Only the wrapped layer bypasses IPEX/oneDNN at apply-time; all other
    layers in the model keep whatever IPEX previously installed.

    Falls back to the original quant_method if qweight is absent (e.g. a
    layer we deliberately skip, or quantization failed).
    """

    def __init__(self, fallback, shape: tuple[int, int], layer_name: str):
        self._fallback = fallback
        self._N, self._K = shape
        self._layer_name = layer_name

    def create_weights(self, *args, **kwargs):
        return self._fallback.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self._fallback, "process_weights_after_loading"):
            try:
                self._fallback.process_weights_after_loading(layer)
            except Exception as e:  # pragma: no cover — fallback path
                logger.debug(
                    "[HYBRID-KERNEL] fallback post-load on %s (%s) failed: %s",
                    self._layer_name, type(self._fallback).__name__, e)

    def apply(self, layer, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = getattr(layer, "_vllm_hybrid_qweight", None)
        if qweight is None:
            return self._fallback.apply(layer, x, bias)

        N = self._N
        K = self._K
        orig_shape = x.shape
        if orig_shape[-1] != K:
            raise RuntimeError(
                f"[HYBRID-KERNEL-Q8_0] layer={self._layer_name} last-dim "
                f"mismatch: x={tuple(orig_shape)} expected K={K}")

        x_2d = x.contiguous().view(-1, K)
        out_2d = torch.empty(x_2d.size(0), N, dtype=x.dtype, device=x.device)

        # bias must be FP32 for current kernel. Upconvert if needed.
        bias_arg = bias
        if bias is not None and bias.dtype != torch.float32:
            bias_arg = bias.to(torch.float32)

        t0 = time.perf_counter() if _trace_enabled() else 0.0
        torch.ops._C_cpu_ops.q8_0_linear(out_2d, x_2d, qweight, bias_arg)
        if _trace_enabled():
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "[HYBRID-KERNEL-Q8_0] layer=%s M=%d N=%d K=%d time=%.3fms",
                self._layer_name, x_2d.size(0), N, K, dt_ms)

        return out_2d.view(*orig_shape[:-1], N)


def _quantize_layer(layer: torch.nn.Module) -> tuple[int, int]:
    """Quantize layer.weight to Q8_0 and attach to layer as
    ``_vllm_hybrid_qweight``. Returns (N, K) of the original weight.
    """
    w = layer.weight.data
    N, K = w.shape
    qw = _quantize_to_q8_0(w)
    layer._vllm_hybrid_qweight = qw
    layer._vllm_hybrid_orig_dtype = w.dtype
    return N, K


def _cpu_ops_available() -> bool:
    try:
        from vllm._custom_ops import HAS_CPU_OPS
    except ImportError:
        return False
    if not HAS_CPU_OPS:
        return False
    if not hasattr(torch.ops, "_C_cpu_ops"):
        return False
    return hasattr(torch.ops._C_cpu_ops, "q8_0_linear") and \
        hasattr(torch.ops._C_cpu_ops, "q8_0_quantize_weight")


def _resolve_architectures(model_config) -> tuple[str, ...]:
    """Best-effort extraction of HF architectures list. Returns empty tuple
    if not available (e.g. tests passing raw nn.Module)."""
    if model_config is None:
        return ()
    hf = getattr(model_config, "hf_config", None)
    if hf is None:
        return ()
    archs = getattr(hf, "architectures", None) or ()
    return tuple(archs)


def _is_target_mlp_module(name: str) -> bool:
    if not any(name.endswith(s) for s in _ALLOWED_SUFFIXES):
        return False
    if any(sub in name for sub in _EXCLUDE_SUBSTRINGS):
        return False
    return True


def patch_mlp_to_q8_0(model: torch.nn.Module,
                      model_config=None,
                      lora_enabled: bool = False) -> int:
    """Replace Qwen2 MLP Linear apply() paths with
    torch.ops._C_cpu_ops.q8_0_linear.

    Scope (Phase A): Qwen2ForCausalLM only. Target modules match
    ``*.mlp.gate_up_proj`` / ``*.mlp.down_proj`` and must not contain any
    MoE-/vision-/audio-specific prefix (see _EXCLUDE_SUBSTRINGS). Only
    the patched layers bypass IPEX at apply-time — the rest of the model
    remains on whatever IPEX installed.

    Returns the number of patched layers (0 on any no-op path).
    """
    if not _env_flag("HYBRID_VNNI_HOT_PATH"):
        logger.info(
            "[HYBRID-KERNEL] §06 disabled (HYBRID_VNNI_HOT_PATH=0)")
        return 0

    if lora_enabled:
        logger.warning(
            "[HYBRID-KERNEL] §06 skipped: LoRA is active. Static Q8_0 "
            "quantization of the base weight is incompatible with runtime "
            "LoRA delta-W. Disable --lora to use this path.")
        return 0

    if not _cpu_ops_available():
        logger.warning(
            "[HYBRID-KERNEL] §06 skipped: _C_cpu_ops / q8_0 op unavailable "
            "(likely AVX-512/VNNI missing in this build).")
        return 0

    archs = _resolve_architectures(model_config)
    if archs and not any(a in _ALLOWED_ARCHS for a in archs):
        logger.warning(
            "[HYBRID-KERNEL] §06 skipped: architecture %s not in allowlist "
            "%s. Extend _ALLOWED_ARCHS after validating the MLP structure.",
            list(archs), sorted(_ALLOWED_ARCHS))
        return 0

    patched = 0
    skipped_filter = 0
    skipped_error = 0
    patched_names: list[str] = []

    for name, module in model.named_modules():
        if not _is_target_mlp_module(name):
            # Silently skip non-targets — they are the vast majority.
            continue
        if not hasattr(module, "quant_method") or module.quant_method is None:
            skipped_filter += 1
            continue
        if not hasattr(module, "weight"):
            skipped_filter += 1
            continue

        try:
            N, K = _quantize_layer(module)
        except ValueError as e:
            logger.warning(
                "[HYBRID-KERNEL] skip layer=%s reason=%s", name, e)
            skipped_error += 1
            continue
        except Exception as e:
            logger.warning(
                "[HYBRID-KERNEL] quantize failed layer=%s reason=%s",
                name, e)
            skipped_error += 1
            continue

        original = module.quant_method
        module.quant_method = _Q8_0LinearMethod(original, (N, K), name)
        module._vllm_hybrid_layer_name = name
        patched += 1
        patched_names.append(name)
        logger.debug(
            "[HYBRID-KERNEL] patch layer=%s shape=[%d,%d] qbytes=%d "
            "fallback=%s",
            name, N, K, module._vllm_hybrid_qweight.numel(),
            type(original).__name__)

    arch_tag = ",".join(archs) if archs else "unknown"
    logger.info(
        "[HYBRID-KERNEL] §06 patched=%d skipped=%d (filter=%d, error=%d) "
        "arch=%s lora=%s scope=Qwen2_MLP(.mlp.gate_up_proj,.mlp.down_proj) "
        "quantize=load-time-1x repack=0 non_patched_layers=ipex_unchanged",
        patched, skipped_filter + skipped_error,
        skipped_filter, skipped_error, arch_tag, lora_enabled)
    if patched_names and _env_flag("VLLM_HYBRID_KERNEL_TRACE"):
        logger.info("[HYBRID-KERNEL] patched list: %s",
                    ", ".join(patched_names))
    return patched
