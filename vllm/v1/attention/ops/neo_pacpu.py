# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO pacpu (CPU paged attention) Python wrapper.

Loads the NEO-built model-specific shared library
(``libpacpu-<model>-tp<TP>.so``) and exposes ``forward_attention`` —
a Python facade over ``torch.ops.pacpu.paged_attention_cpu`` that
handles vLLM's per-layer KV cache layout ↔ NEO's multi-layer format.

NEO upstream: https://github.com/NEO-MLSys25/NEO (MLSys 2025, Apache 2.0).
This module is a *thin adapter* — the kernel itself is NEO's pacpu/
(ISPC + C++) cherry-picked under ``csrc/cpu/pacpu/``.

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §7.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Build artifact lookup — ``csrc/cpu/pacpu/build/libpacpu-<m>-tp<n>.so``
# ----------------------------------------------------------------------
_PACPU_BUILD_DIR = (
    Path(__file__).resolve().parents[4]    # vllm_hybrid/
    / "csrc" / "cpu" / "pacpu" / "build"
)

# Model name → NEO build macro (must match dtype.h's #ifdef branches).
_MODEL_TO_NEO_MACRO: dict[str, str] = {
    "qwen2.5-1.5b": "qwen2_5_1_5b",
    "qwen/qwen2.5-1.5b-instruct": "qwen2_5_1_5b",
    "llama-3.3-70b": "llama3_3_70b",
    "meta-llama/llama-3.3-70b-instruct": "llama3_3_70b",
    "llama-3.1-70b": "llama3_3_70b",     # same dim-set as 3.3
    "llama-3-70b":   "llama3_3_70b",
    "llama-2-70b":   "llama3_3_70b",
}

_loaded_libraries: dict[str, str] = {}   # macro → loaded .so path


def _resolve_neo_macro(model_name_or_path: str) -> str | None:
    """Best-effort mapping from vLLM ``model_config.model`` (a path or HF
    id) to the NEO build macro. Returns ``None`` if no match."""
    needle = model_name_or_path.lower()
    for key, macro in _MODEL_TO_NEO_MACRO.items():
        if key in needle:
            return macro
    return None


def _so_path(model_macro: str, tp_degree: int) -> Path:
    return _PACPU_BUILD_DIR / f"libpacpu-{model_macro}-tp{tp_degree}.so"


def is_kernel_available(model_name: str, tp_degree: int) -> bool:
    """``True`` iff a pre-built pacpu shared library exists for the
    given model + TP. Caller should fall back to the pure-Python
    reference (``cpu_attention.forward_attention``) when ``False``."""
    macro = _resolve_neo_macro(model_name)
    if macro is None:
        return False
    return _so_path(macro, tp_degree).is_file()


def load_kernel(model_name: str, tp_degree: int) -> bool:
    """Load the model-specific pacpu shared library exactly once.
    Subsequent calls are no-ops. Returns True if successfully loaded
    (or already loaded)."""
    macro = _resolve_neo_macro(model_name)
    if macro is None:
        logger.warning(
            "neo_pacpu: no NEO build macro mapping for model %r — "
            "fall back to pure-Python reference.",
            model_name,
        )
        return False
    if macro in _loaded_libraries:
        return True

    so = _so_path(macro, tp_degree)
    if not so.is_file():
        logger.warning(
            "neo_pacpu: pre-built %s not found. Build with:\n"
            "    cd csrc/cpu/pacpu && bash build.sh %s %d\n"
            "(needs g++-12 and ispc on PATH). Falling back to "
            "pure-Python reference.",
            so, macro, tp_degree,
        )
        return False
    try:
        torch.ops.load_library(str(so))
    except Exception as e:  # noqa: BLE001
        logger.warning("neo_pacpu: torch.ops.load_library(%s) failed: %s",
                       so, e)
        return False
    _loaded_libraries[macro] = str(so)
    logger.info("neo_pacpu: loaded %s", so)
    return True


# ----------------------------------------------------------------------
# vLLM ↔ NEO KV layout adapter
# ----------------------------------------------------------------------
# vLLM HND per-layer KV: (num_blocks, num_kv_heads, block_size, head_dim)
# NEO multi-layer KV  : (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
#
# We treat each per-layer view as ``num_layers=1`` and feed ``cur_layer=0``.
# vLLM NHD layout: (num_blocks, block_size, num_kv_heads, head_dim) — needs
# ``.permute(0, 2, 1, 3)`` to get HND. Permute is a *view*; ISPC requires
# contiguous input, so we ``.contiguous()`` only when stride mismatch.


def _to_neo_kv_view(
    kv_layer: torch.Tensor,    # vLLM per-layer K or V
) -> torch.Tensor:
    """Reshape vLLM per-layer KV to NEO's expected
    ``(num_layers=1, num_blocks, num_kv_heads, block_size, head_dim)``
    contiguous tensor.

    NEO's pacpu uses raw pointer arithmetic on a flat buffer — a view
    is enough as long as the strides match a fully contiguous
    ``(num_layers, num_blocks, num_kv_heads, block_size, head_dim)``
    layout. When vLLM is configured with NHD layout we must ``.contiguous()``
    after the permute (slow path); HND layout is zero-copy.
    """
    if kv_layer.dim() == 4:
        # (num_blocks, num_kv_heads, block_size, head_dim) — HND, OK
        # OR (num_blocks, block_size, num_kv_heads, head_dim) — NHD, permute
        # Heuristic: if dim 1 size == dim 2 size we can't tell; rely on
        # caller passing HND (vLLM exposes this via get_kv_cache_layout()).
        layered = kv_layer.unsqueeze(0)         # add num_layers=1
    elif kv_layer.dim() == 5:
        layered = kv_layer                       # already has layer dim
    else:
        raise ValueError(
            f"_to_neo_kv_view: unexpected kv_layer.dim()={kv_layer.dim()}"
        )

    if not layered.is_contiguous():
        layered = layered.contiguous()
    return layered


# ----------------------------------------------------------------------
# Python facade
# ----------------------------------------------------------------------
def forward_attention(
    *,
    cur_layer: int,
    softmax_scale: float,
    seq_ids: list[int],
    seq_lengths: list[int],
    q: torch.Tensor,           # (batch_size, num_q_heads, head_dim) FP16
    k_new: torch.Tensor,       # (batch_size, num_kv_heads, head_dim) — current step's KV
    v_new: torch.Tensor,
    k_cache_layer: torch.Tensor,    # vLLM per-layer K cache (HND-shaped)
    v_cache_layer: torch.Tensor,    # vLLM per-layer V cache (HND-shaped)
    block_table: torch.Tensor,      # (batch_size, max_blocks_per_seq) int32
    output: torch.Tensor,           # (batch_size, num_q_heads * head_dim) FP32 — filled in-place
) -> None:
    """Call NEO's ``paged_attention_cpu`` after view-aliasing the
    per-layer KV to the multi-layer layout NEO expects.

    The kernel writes the result into ``output`` in-place. Caller is
    responsible for moving the result to GPU (output → q's hidden states).
    """
    if not hasattr(torch.ops, "pacpu"):
        raise RuntimeError(
            "neo_pacpu: torch.ops.pacpu not registered — "
            "did you call load_kernel() first?"
        )

    # NEO indexes K/V via cur_layer + num_layers; we model per-layer view
    # as num_layers=1 and pass cur_layer=0.
    k_neo = _to_neo_kv_view(k_cache_layer)
    v_neo = _to_neo_kv_view(v_cache_layer)

    torch.ops.pacpu.paged_attention_cpu(
        0,                       # cur_layer (always 0 in per-layer view)
        float(softmax_scale),
        list(seq_ids),
        list(seq_lengths),
        q,
        k_new,
        v_new,
        k_neo,
        v_neo,
        block_table,
        output,
    )
