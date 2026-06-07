# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KT-Kernel binding for vLLM MoE expert CPU offload (PoC, SUB_201).

This module provides a thin wrapper around kt_kernel's KTMoEWrapper so that
vLLM's UnquantizedFusedMoEMethod can offload a portion of the MoE experts to
CPU (AMX BF16). It is enabled by ``VLLM_MOE_CPU_OFFLOAD=1``.

The integration mirrors SGLang's ``KTEPWrapperMethod.apply`` pattern:
  1. (cpu_stream) wrapper.submit_forward(x_staging, topk_ids_full, topk_weights, stream)
  2. (main_stream) gpu_method.apply(x, masked_topk_ids, ...) where CPU experts -> -1
  3. (cpu_stream) cpu_out_on_gpu = wrapper.sync_forward(x_staging, stream)
  4. (main_stream) output = gpu_out + cpu_out_on_gpu

Origin: kt-sglang 耦合 (see /workspace/sglang_kt_prj/.../kt_ep_wrapper.py).

Protection mechanisms (#34 fix for CUDA illegal memory access at layer 5+):

* ``SharedStagingBuffer``: process-wide single GPU staging buffer of shape
  ``(max_tokens, hidden_size)`` shared across all MoE layers. Mirrors SGLang's
  ``SharedStagingBuffer`` (kt_ep_wrapper.py:113). Avoids ephemeral
  ``x.contiguous().clone()`` per call, which under PyTorch's stream-aware
  caching allocator can be freed while kt-kernel's C++ ``submit_with_cuda_stream``
  task still holds the raw pointer.
* ``register_kt_capture_batch_sizes(...)``: registers batch sizes with
  ``KExpertsCPUBuffer.capture_bs`` so the corresponding ``(input_tensor_cpu,
  output_cpu, ...)`` tuples land in the ``capture_buffers`` dict (process
  lifetime) instead of the single ``temp_buffer`` slot. This eliminates the
  cross-layer race in which layer N's submit_forward task points at the
  ``temp_buffer`` slot that layer N+1's ``get_buffer`` overwrites when the
  call-site batch size changes (e.g. profile_run uses max_num_tokens but
  individual decode steps use smaller batches).
* Dynamic registration on first sighting in ``_kt_forward``: enforces the
  capture-bs guarantee even when the engine surprises us with a batch size
  not anticipated at attach time (enforce-eager mode lacks the
  cuda_graph_runner's ``capture_bs`` list).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:  # pragma: no cover
    from kt_kernel import KTMoEWrapper  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False", "no")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _str_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def is_cpu_offload_enabled() -> bool:
    """Return True if VLLM_MOE_CPU_OFFLOAD=1."""
    return _bool_env("VLLM_MOE_CPU_OFFLOAD", "0")


def cpu_offload_config() -> dict[str, Any]:
    """Read the PoC env-var-driven config.

    Knobs:
      * ``VLLM_MOE_NUM_GPU_EXPERTS``  – count of GPU-resident experts (default 32).
      * ``VLLM_MOE_CPUINFER_THREADS`` – kt CPU threads (default 112).
      * ``VLLM_MOE_THREADPOOL_COUNT`` – kt NUMA subpools (default 2).
      * ``VLLM_MOE_KT_METHOD``        – BF16 / AMXINT8 / LLAMAFILE (default BF16).
      * ``VLLM_MOE_KT_WEIGHT_PATH``   – HF model dir containing safetensors. Required.
      * ``VLLM_MOE_MAX_DEFERRED``     – kt max_deferred_experts_per_token (default 0).
      * ``VLLM_MOE_CHUNKED_PREFILL``  – kt chunked_prefill_size (default 8192).
      * ``VLLM_MOE_KT_SITE_PACKAGES`` – path to kt_kernel site-packages
                                        (default /workspace/sglang_kt_prj/lib/python3.12/site-packages).
    """
    return {
        "num_gpu_experts": _int_env("VLLM_MOE_NUM_GPU_EXPERTS", 32),
        "cpuinfer_threads": _int_env("VLLM_MOE_CPUINFER_THREADS", 112),
        "threadpool_count": _int_env("VLLM_MOE_THREADPOOL_COUNT", 2),
        "kt_method": _str_env("VLLM_MOE_KT_METHOD", "BF16"),
        "weight_path": _str_env("VLLM_MOE_KT_WEIGHT_PATH", ""),
        "max_deferred": _int_env("VLLM_MOE_MAX_DEFERRED", 0),
        "chunked_prefill": _int_env("VLLM_MOE_CHUNKED_PREFILL", 8192),
        "kt_site_packages": _str_env(
            "VLLM_MOE_KT_SITE_PACKAGES",
            "/workspace/sglang_kt_prj/lib/python3.12/site-packages",
        ),
    }


def ensure_kt_kernel_importable() -> bool:
    """Make ``kt_kernel`` importable without poisoning the rest of sys.path.

    Naively prepending the kt-kernel site-packages would shadow the project's
    transformers/torch versions and break vLLM's tokenizer initialization
    (e.g. ``standardize_rope_params`` was added in transformers >= 5.x, while
    the project ships 4.57.x). Instead we append at the end so vLLM's own
    packages keep priority, and we eagerly import ``kt_kernel`` so its top-level
    side-effects (loading the AMX extension) only run once.
    """
    try:
        import kt_kernel  # noqa: F401
        return True
    except ImportError:
        pass

    cfg = cpu_offload_config()
    p = cfg["kt_site_packages"]
    if p and os.path.isdir(p) and p not in sys.path:
        # Append, not insert, to preserve vllm_dev_prj's package priority.
        sys.path.append(p)
        logger.warning("[kt-binding] appended sys.path (low priority): %s", p)
    try:
        import kt_kernel  # noqa: F401
        return True
    except ImportError as e:
        logger.error("[kt-binding] kt_kernel import failed: %s", e)
        return False


def build_gpu_experts_mask(num_experts: int, num_gpu_experts: int) -> torch.Tensor:
    """Build a [num_experts] bool mask. First num_gpu_experts are GPU."""
    mask = torch.zeros(num_experts, dtype=torch.bool)
    n = max(0, min(num_gpu_experts, num_experts))
    if n > 0:
        mask[:n] = True
    return mask


def build_logical_to_gpu_index(mask: torch.Tensor) -> torch.Tensor:
    """For each global expert id, return its local GPU index (0..N-1) or -1."""
    n_total = mask.shape[0]
    out = torch.full((n_total,), -1, dtype=torch.int32)
    gpu_ids = torch.where(mask)[0]
    out[gpu_ids] = torch.arange(len(gpu_ids), dtype=torch.int32)
    return out


def mask_and_remap_topk(
    topk_ids: torch.Tensor,
    gpu_experts_mask_cuda: torch.Tensor,
    logical_to_gpu_index_cuda: torch.Tensor,
    invalid_sentinel: int,
) -> torch.Tensor:
    """Mask CPU experts and remap GPU experts to GPU-local weight indices.

    For each (token, k) slot:
      * if expert is GPU-resident → write the local GPU index in [0,
        num_gpu_experts).
      * else (CPU expert) → write ``invalid_sentinel`` (caller supplies
        ``num_total_experts``, i.e. the GLOBAL count, so vLLM's CUDA
        ``moe_align_block_size_kernel`` short-circuits via its
        ``if (expert_id >= num_experts) continue`` guard).

    Why not ``-1`` like SGLang? vLLM's ``moe_align_block_size_kernel``
    (csrc/moe/moe_align_sum_kernels.cu:126) only guards
    ``expert_id >= num_experts`` unless ``has_expert_map`` is true; a
    raw ``-1`` would underflow ``warp_idx = expert_id / experts_per_warp``
    into a negative shared-memory offset and write OOB → CUDA illegal
    memory access (root cause #34). Using ``num_total_experts`` is the
    OOB-safe sentinel for the no-expert-map call shape we use.
    """
    is_gpu = gpu_experts_mask_cuda[topk_ids]
    remap = logical_to_gpu_index_cuda[topk_ids]
    sentinel = torch.full_like(topk_ids, int(invalid_sentinel))
    return torch.where(is_gpu, remap.to(topk_ids.dtype), sentinel)


# ---------------------------------------------------------------------------
# Process-wide protection mechanisms (#34 fix for CUDA illegal memory access)
# ---------------------------------------------------------------------------

_SHARED_STAGING_BUFFER: Optional["SharedStagingBuffer"] = None
_REGISTERED_CAPTURE_BS: set[int] = set()
_KT_LOCK = threading.Lock()


class SharedStagingBuffer:
    """Process-wide GPU staging buffer shared by all MoE layers.

    Mirrors SGLang's ``SharedStagingBuffer`` (kt_ep_wrapper.py:113). A single
    contiguous ``(max_tokens, hidden_size)`` BF16 buffer lives for the entire
    process so the kt-kernel C++ ``submit_with_cuda_stream`` task has a
    stable raw GPU pointer that survives the cross-stream gap until
    ``sync_forward`` retires it. Per-call ``x.clone()`` does not satisfy this
    invariant under PyTorch's stream-aware caching allocator (free can occur
    before the C++ task starts).
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.buffer = torch.empty(
            (max_tokens, hidden_size), dtype=dtype, device=device
        )
        mib = self.buffer.numel() * self.buffer.element_size() / 1024**2
        logger.info(
            "[kt-binding] SharedStagingBuffer: %.1f MiB (max_tokens=%d hidden=%d dtype=%s)",
            mib, max_tokens, hidden_size, str(dtype),
        )

    def get_slice(self, num_tokens: int) -> torch.Tensor:
        assert num_tokens <= self.max_tokens, (
            f"num_tokens={num_tokens} exceeds staging buffer max={self.max_tokens}"
        )
        return self.buffer[:num_tokens]


def get_or_create_shared_staging_buffer(
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SharedStagingBuffer:
    """Get or create the global staging buffer. Idempotent across layers."""
    global _SHARED_STAGING_BUFFER
    with _KT_LOCK:
        if _SHARED_STAGING_BUFFER is None:
            _SHARED_STAGING_BUFFER = SharedStagingBuffer(
                max_tokens=max_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
                device=device,
            )
        else:
            # Grow if needed (rare; only first init usually).
            if _SHARED_STAGING_BUFFER.max_tokens < max_tokens or (
                _SHARED_STAGING_BUFFER.hidden_size != hidden_size
            ):
                logger.warning(
                    "[kt-binding] re-allocating SharedStagingBuffer: "
                    "old=(max=%d,hidden=%d) new=(max=%d,hidden=%d)",
                    _SHARED_STAGING_BUFFER.max_tokens,
                    _SHARED_STAGING_BUFFER.hidden_size,
                    max_tokens, hidden_size,
                )
                _SHARED_STAGING_BUFFER = SharedStagingBuffer(
                    max_tokens=max_tokens,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    device=device,
                )
        return _SHARED_STAGING_BUFFER


def register_kt_capture_batch_sizes(batch_sizes: list[int]) -> None:
    """Register batch sizes with kt-kernel's KExpertsCPUBuffer capture cache.

    Equivalent to SGLang's ``KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)``
    call in cuda_graph_runner.py:497. Each registered batch size's CPU
    pinned-memory tuple goes into the ``capture_buffers`` dict (process
    lifetime) instead of the single ``temp_buffer`` slot that gets overwritten
    on every new batch shape.

    Safe to call multiple times; the final list is the union of all calls.
    """
    if not ensure_kt_kernel_importable():
        return
    from kt_kernel import KTMoEWrapper  # type: ignore[import-not-found]
    with _KT_LOCK:
        new_bs = [b for b in batch_sizes if b not in _REGISTERED_CAPTURE_BS]
        if not new_bs:
            return
        _REGISTERED_CAPTURE_BS.update(new_bs)
        merged = sorted(_REGISTERED_CAPTURE_BS)
    KTMoEWrapper.set_capture_batch_sizes(merged)
    logger.info(
        "[kt-binding] registered capture batch sizes (+%s) -> total %s",
        new_bs, merged,
    )


def ensure_batch_size_captured(batch_size: int) -> None:
    """Idempotently ensure ``batch_size`` is in the kt capture set.

    Called from ``_kt_forward`` on first sighting of a batch_size. Cheap
    after the first call (set-membership). vLLM in enforce-eager mode does
    not pre-publish a capture_bs list, so this fills the gap.
    """
    if batch_size in _REGISTERED_CAPTURE_BS:
        return
    register_kt_capture_batch_sizes([batch_size])


class KtKernelLayerWrapper:
    """Per-layer wrapper holding a kt_kernel.KTMoEWrapper + helper tensors.

    Construction is lazy: the actual KTMoEWrapper is built on the first
    ``ensure_initialized()`` call so that we run after CUDA is up and the
    weight path is known.
    """

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        device: torch.device,
        weight_path: str,
        num_gpu_experts: int,
        cpuinfer_threads: int,
        threadpool_count: int,
        method: str,
        max_deferred: int,
        chunked_prefill: int,
    ) -> None:
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.device = device
        self.weight_path = weight_path
        self.num_gpu_experts = num_gpu_experts
        self.cpuinfer_threads = cpuinfer_threads
        self.threadpool_count = threadpool_count
        self.method = method
        self.max_deferred = max_deferred
        self.chunked_prefill = chunked_prefill

        self.gpu_experts_mask_cpu = build_gpu_experts_mask(num_experts, num_gpu_experts)
        self.logical_to_gpu_index_cpu = build_logical_to_gpu_index(
            self.gpu_experts_mask_cpu
        )

        self.gpu_experts_mask_cuda: Optional[torch.Tensor] = None
        self.logical_to_gpu_index_cuda: Optional[torch.Tensor] = None
        # ``expert_map`` is the canonical vLLM way to tell its fused_moe
        # triton kernel "CPU experts -> -1, GPU experts -> local idx". Built
        # once during ensure_initialized and reused across forwards.
        self.expert_map_cuda: Optional[torch.Tensor] = None

        self._kt_wrapper: Optional[Any] = None  # KTMoEWrapper
        self._cpu_stream: Optional[torch.cuda.Stream] = None
        self._sync_done_event: Optional[torch.cuda.Event] = None
        self._weights_loaded = False

    def ensure_initialized(self) -> None:
        if self._kt_wrapper is not None:
            return
        if not ensure_kt_kernel_importable():
            raise RuntimeError(
                "kt_kernel not importable — set VLLM_MOE_KT_SITE_PACKAGES "
                "to the path containing kt_kernel package."
            )
        from kt_kernel import KTMoEWrapper  # type: ignore[import-not-found]

        if not self.weight_path:
            raise RuntimeError(
                "VLLM_MOE_KT_WEIGHT_PATH must be set (HF model dir with safetensors)."
            )

        # Move mask + index to GPU
        self.gpu_experts_mask_cuda = self.gpu_experts_mask_cpu.to(self.device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index_cpu.to(self.device)
        # expert_map[i] = GPU-local index if GPU, else -1. Matches vLLM's
        # FusedMoE.expert_map semantics so we can pass topk_ids unchanged.
        self.expert_map_cuda = self.logical_to_gpu_index_cuda.clone().to(torch.int32)

        # Build the kt-kernel wrapper
        logger.info(
            "[kt-binding] init layer %d: num_experts=%d top_k=%d num_gpu_experts=%d "
            "hidden=%d inter=%d method=%s threads=%d pools=%d",
            self.layer_idx,
            self.num_experts,
            self.num_experts_per_tok,
            self.num_gpu_experts,
            self.hidden_size,
            self.moe_intermediate_size,
            self.method,
            self.cpuinfer_threads,
            self.threadpool_count,
        )
        self._kt_wrapper = KTMoEWrapper(
            layer_idx=self.layer_idx,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            gpu_experts_mask=self.gpu_experts_mask_cpu,
            cpuinfer_threads=self.cpuinfer_threads,
            threadpool_count=self.threadpool_count,
            weight_path=self.weight_path,
            chunked_prefill_size=self.chunked_prefill,
            method=self.method,
            max_deferred_experts_per_token=self.max_deferred,
            mode="inference",
        )

        # CPU stream + sync event
        self._cpu_stream = torch.cuda.Stream(device=self.device)
        self._sync_done_event = torch.cuda.Event()

        # SharedStagingBuffer (process-wide). Sized for the worst-case batch
        # vLLM will ever feed into MoE: ``chunked_prefill`` (== profile_run's
        # max_num_tokens). Created once per process, shared across all layers.
        max_tokens = max(self.chunked_prefill, 1)
        get_or_create_shared_staging_buffer(
            max_tokens=max_tokens,
            hidden_size=self.hidden_size,
            dtype=torch.bfloat16,  # kt-kernel CPU path consumes BF16
            device=self.device,
        )

    def load_weights(self, physical_to_logical_map_cpu: Optional[torch.Tensor] = None) -> None:
        """Load expert weights from disk into kt-kernel's shared memory."""
        if self._weights_loaded:
            return
        self.ensure_initialized()
        if physical_to_logical_map_cpu is None:
            physical_to_logical_map_cpu = torch.arange(
                self.num_experts, dtype=torch.long, device="cpu"
            )
        logger.info("[kt-binding] loading weights for layer %d (path=%s)",
                    self.layer_idx, self.weight_path)
        self._kt_wrapper.load_weights(physical_to_logical_map_cpu)
        self._weights_loaded = True

    @property
    def cpu_stream(self) -> torch.cuda.Stream:
        assert self._cpu_stream is not None
        return self._cpu_stream

    @property
    def sync_done_event(self) -> torch.cuda.Event:
        assert self._sync_done_event is not None
        return self._sync_done_event

    @property
    def kt_wrapper(self) -> Any:
        assert self._kt_wrapper is not None
        return self._kt_wrapper

    def is_ready(self) -> bool:
        return self._kt_wrapper is not None and self._weights_loaded
