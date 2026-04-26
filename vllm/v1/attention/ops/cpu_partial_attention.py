# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cold-KV CPU Partial Attention — Python wrapper + reference impl.

Implements the §4.1 Python reference and the §4.3 user-facing wrapper
of TSK_001. The C++ paths (4.2a AVX-512, 4.2b AMX, 4.2c portable) are
added later — until then the wrapper dispatches everything to the
reference path.

See ``shadow_assists/features/IDE_006/TSK_001.md`` for the spec.

Notation
--------
- ``num_seqs``       : number of sequences in the batch.
- ``num_tokens``     : total query tokens across the batch.
- ``num_q_heads``    : query heads (e.g. 32 for Qwen2.5-7B).
- ``num_kv_heads``   : KV heads, can be smaller for GQA (e.g. 4).
- ``head_dim``       : per-head dimension (e.g. 128).
- ``block_size``     : tokens per paged KV block.

LSE merge math: arXiv 2501.01005 §2.2.
"""

from __future__ import annotations

import math
import os
from enum import Enum
from typing import Optional

import torch

from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


__all__ = [
    "ISAPath",
    "select_isa_path",
    "forward_partial_with_lse",
    "python_reference_partial_attention",
]


# ---------------------------------------------------------------------
# §4.3 dispatch — 4-stage policy:
#   AMX → AVX-512 → portable (C++) → Python reference
# Until the C++ kernels land, only the Python reference is wired.
# ---------------------------------------------------------------------


class ISAPath(str, Enum):
    AMX = "amx"
    AVX512 = "avx512"
    PORTABLE = "portable"
    PYTHON_REF = "python_ref"


def _has_amx() -> bool:
    """True if the current CPU exposes AMX BF16 instructions."""
    try:
        import cpuinfo  # type: ignore
    except Exception:
        return False
    info = cpuinfo.get_cpu_info()
    flags = info.get("flags", [])
    return "amx_bf16" in flags or "amx_tile" in flags


def _has_avx512() -> bool:
    """True if the current CPU exposes AVX-512 (foundation)."""
    try:
        import cpuinfo  # type: ignore
    except Exception:
        return False
    info = cpuinfo.get_cpu_info()
    flags = info.get("flags", [])
    return any(f.startswith("avx512f") for f in flags)


def select_isa_path() -> ISAPath:
    """Return the highest-priority available ISA path on this machine.

    Order: AMX → AVX-512 → portable (always available C++) →
    Python reference. Each step is gated on (a) the corresponding kernel
    being loaded successfully and (b) the host CPU exposing the ISA.
    """
    if _has_amx_kernel() and _has_amx():
        return ISAPath.AMX
    if _has_avx512_kernel() and _has_avx512():
        return ISAPath.AVX512
    if _has_portable_kernel():
        return ISAPath.PORTABLE
    return ISAPath.PYTHON_REF


# ---------------------------------------------------------------------
# C++ JIT loaders (TSK_001 §4.2c portable; AVX-512 / AMX added later).
#
# Until AVX-512 / AMX kernels are wired, only the portable kernel is
# JIT-compiled here. The two helpers below stay False so that the
# dispatch correctly falls back to portable / Python ref.
# ---------------------------------------------------------------------


_PORTABLE_MOD = None
_PORTABLE_LOAD_ERROR: Optional[Exception] = None
_PORTABLE_LOAD_ATTEMPTED = False


def _portable_source_path() -> str:
    # vllm/v1/attention/ops/cpu_partial_attention.py → repo root is 5 up.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.join(
        repo_root, "csrc", "cpu", "partial_attention_portable.cpp",
    )


def _try_load_portable():
    """Lazily JIT-compile the portable C++ kernel.

    First call takes ~30-60s (compilation); subsequent calls hit the
    PyTorch extension cache. On compilation failure we record the
    exception and stop trying — :func:`select_isa_path` then returns
    :data:`ISAPath.PYTHON_REF`.
    """
    global _PORTABLE_MOD, _PORTABLE_LOAD_ERROR, _PORTABLE_LOAD_ATTEMPTED
    if _PORTABLE_LOAD_ATTEMPTED:
        return
    _PORTABLE_LOAD_ATTEMPTED = True
    try:
        from torch.utils.cpp_extension import load
        src = _portable_source_path()
        if not os.path.isfile(src):
            raise FileNotFoundError(f"portable kernel source missing: {src}")
        _PORTABLE_MOD = load(
            name="vllm_partial_attention_portable",
            sources=[src],
            extra_cflags=["-O3", "-ftree-vectorize", "-fno-strict-aliasing"],
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        _PORTABLE_LOAD_ERROR = e
        _PORTABLE_MOD = None


def _has_portable_kernel() -> bool:
    _try_load_portable()
    return _PORTABLE_MOD is not None


def _has_avx512_kernel() -> bool:
    # AVX-512 C++ kernel is part of TSK_001 §4.2a (Phase 2 prod).
    return False


def _has_amx_kernel() -> bool:
    # AMX C++ kernel is part of TSK_001 §4.2b (Phase 2 prod).
    return False


# ---------------------------------------------------------------------
# §4.1 Python reference — torch ops, LSE-returning
# ---------------------------------------------------------------------


def python_reference_partial_attention(
    *,
    query: torch.Tensor,                # [num_tokens, num_q_heads, head_dim]
    cold_kv_cache: torch.Tensor,        # [num_blocks, page_size_bytes] int8 — combined,
                                        # OR [num_blocks, kv_block_bytes] int8 — K-only
                                        # when cold_kv_cache_v is provided (split mode).
    cold_kv_layout: KVPageLayout,       # how K/V are laid out inside a page
    cold_block_ids: torch.Tensor,       # [num_seqs, max_cold_blocks_per_seq]
    cold_block_lens: torch.Tensor,      # [num_seqs]
    cu_seqlens_q: torch.Tensor,         # [num_seqs + 1]
    seq_lens_total: torch.Tensor,       # [num_seqs]
    query_positions: torch.Tensor,      # [num_tokens]
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    cold_kv_cache_v: Optional[torch.Tensor] = None,  # [num_blocks, kv_block_bytes]
                                                      # int8 — V-only, paired with K-only
                                                      # cold_kv_cache for split-K/V layout
                                                      # (e.g. FlashAttention's
                                                      # OffloadingConnector mirror).
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute partial attention over the cold KV blocks, returning
    ``(O_cold, LSE_cold)``.

    This is the slow but numerically stable Python reference. SIMD /
    AMX paths are validated against this implementation (TST_001 단계 B).

    Returns
    -------
    O_cold:
        ``[num_tokens, num_q_heads, head_dim]``, same dtype as ``query``.
    LSE_cold:
        ``[num_q_heads, num_tokens]``, ``float32``. Layout matches
        :func:`vllm.v1.attention.ops.merge_attn_states.merge_attn_states`'s
        ``prefix_lse`` / ``suffix_lse`` argument.

    For sequences with no cold blocks the partial output is zero and
    the LSE is ``-inf`` — when merged via online softmax this naturally
    drops out.
    """
    layout = cold_kv_layout
    num_tokens, num_q_heads, head_dim = query.shape
    if head_dim != layout.head_dim:
        raise ValueError(
            f"query head_dim {head_dim} != layout head_dim "
            f"{layout.head_dim}"
        )
    num_seqs = cu_seqlens_q.shape[0] - 1

    # Reinterpret canonical int8 cold KV as typed (num_blocks, block_size,
    # num_kv_heads, head_dim) views. Combined mode treats cold_kv_cache as
    # [num_blocks, K_bytes + V_bytes]; split mode reads K from
    # cold_kv_cache and V from cold_kv_cache_v separately (matches FA's
    # OffloadingConnector mirror layout).
    if cold_kv_cache_v is None:
        adapter = KVViewAdapter(cold_kv_cache, layout)
    else:
        adapter = KVViewAdapter.from_split_kv(
            cold_kv_cache, cold_kv_cache_v, layout
        )
    K_blocks = adapter.k_view()  # [num_blocks, B, num_kv_heads, head_dim]
    V_blocks = adapter.v_view()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # GQA: number of query heads per KV head must be integer.
    if num_q_heads % layout.num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads {num_q_heads} not divisible by "
            f"num_kv_heads {layout.num_kv_heads}"
        )
    q_per_kv = num_q_heads // layout.num_kv_heads

    O_cold = torch.zeros_like(query)
    LSE_cold = torch.full(
        (num_q_heads, num_tokens), float("-inf"),
        dtype=torch.float32, device=query.device,
    )

    # ---- per-sequence inner loop (correctness > speed) -----------
    for s in range(num_seqs):
        q_start = int(cu_seqlens_q[s].item())
        q_end = int(cu_seqlens_q[s + 1].item())
        n_cold_blocks = int(cold_block_lens[s].item())
        if q_end <= q_start or n_cold_blocks <= 0:
            continue

        # Gather this sequence's cold K/V into contiguous tensors.
        block_ids_s = cold_block_ids[s, :n_cold_blocks].to(torch.long)
        K_s = K_blocks.index_select(0, block_ids_s)  # [n_cold, B, kv_h, hd]
        V_s = V_blocks.index_select(0, block_ids_s)
        # Flatten to per-token KV: [n_cold * B, num_kv_heads, head_dim]
        K_s = K_s.reshape(-1, layout.num_kv_heads, head_dim)
        V_s = V_s.reshape(-1, layout.num_kv_heads, head_dim)

        # GQA broadcast: expand to num_q_heads.
        # K_s, V_s : [n_cold_kv_tokens, num_q_heads, head_dim]
        K_s = K_s.repeat_interleave(q_per_kv, dim=1)
        V_s = V_s.repeat_interleave(q_per_kv, dim=1)

        n_cold_kv_tokens = K_s.shape[0]

        Q_s = query[q_start:q_end]  # [q_len, num_q_heads, head_dim]

        # Masking: a cold KV token at absolute position ``p`` is valid
        # for query token at absolute position ``q`` iff p <= q and the
        # block is not beyond the sequence's KV length.
        # Cold block IDs index into K_blocks; we assume the caller has
        # already filtered to "cold" blocks (= lower part of the seq's
        # KV) so positions 0..n_cold_kv_tokens-1 are the absolute KV
        # positions of these tokens.
        cold_kv_positions = torch.arange(
            n_cold_kv_tokens, device=query.device,
        )
        q_positions_s = query_positions[q_start:q_end]  # [q_len]

        # Compute attention: [q_len, num_q_heads, n_cold_kv_tokens]
        # scores[t, h, k] = (Q[t, h, :] · K[k, h, :]) * scale
        scores = torch.einsum(
            "thd,khd->thk",
            Q_s.float(), K_s.float(),
        ) * softmax_scale

        if causal:
            # mask[t, k] = q_positions_s[t] >= cold_kv_positions[k]
            mask = q_positions_s.unsqueeze(1) < cold_kv_positions.unsqueeze(0)
            # Broadcast over heads dim.
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        # Online softmax over K dim; LSE returned in float32.
        # If a row is all -inf (no valid cold token), we leave the row
        # at LSE = -inf and O = 0 by clamping.
        max_scores = scores.amax(dim=-1, keepdim=True)
        # Replace -inf max with 0 to keep the subtraction well-defined;
        # rows that stay -inf will be re-masked to 0 below.
        valid_max = max_scores.masked_fill(
            torch.isinf(max_scores), 0.0,
        )
        exp_scores = (scores - valid_max).exp()
        # zero out rows where all scores were -inf
        invalid_row = torch.isinf(max_scores).squeeze(-1)
        if invalid_row.any():
            exp_scores = exp_scores.masked_fill(
                invalid_row.unsqueeze(-1), 0.0,
            )
        sum_exp = exp_scores.sum(dim=-1)               # [q_len, num_q_heads]
        # Probs: [q_len, num_q_heads, n_cold_kv_tokens]
        probs = exp_scores / sum_exp.unsqueeze(-1).clamp(min=1e-30)

        O_s = torch.einsum(
            "thk,khd->thd",
            probs, V_s.float(),
        ).to(query.dtype)
        O_cold[q_start:q_end] = torch.where(
            invalid_row.unsqueeze(-1),
            torch.zeros_like(O_s),
            O_s,
        )

        # LSE = max + log(sum_exp). Layout: [num_q_heads, num_tokens].
        lse_s = (
            valid_max.squeeze(-1) + sum_exp.clamp(min=1e-30).log()
        )  # [q_len, num_q_heads]
        lse_s = lse_s.transpose(0, 1)  # [num_q_heads, q_len]
        # Where the row was all -inf, restore -inf in LSE.
        lse_s = lse_s.masked_fill(
            invalid_row.transpose(0, 1) if invalid_row.dim() > 1
            else invalid_row.unsqueeze(0).expand(num_q_heads, -1),
            float("-inf"),
        )
        LSE_cold[:, q_start:q_end] = lse_s

    return O_cold, LSE_cold


# ---------------------------------------------------------------------
# §4.3 Python wrapper / dispatch entry point
# ---------------------------------------------------------------------


def forward_partial_with_lse(
    *,
    query: torch.Tensor,
    cold_kv_cache: torch.Tensor,
    cold_kv_layout: KVPageLayout,
    cold_block_ids: torch.Tensor,
    cold_block_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_total: torch.Tensor,
    query_positions: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    cold_kv_cache_v: Optional[torch.Tensor] = None,
    _force_path: Optional[ISAPath] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """User-facing entry point for cold-KV CPU partial attention.

    Selects the best available ISA path (or honours ``_force_path``)
    and returns ``(O_cold, LSE_cold)``. See the module docstring for
    shape conventions.

    The C++ paths are not yet wired; until they are, this dispatches to
    the Python reference for any path. The dispatch table itself is in
    place so the eventual switch becomes a one-line change in
    :func:`select_isa_path` and the dispatch table below.
    """
    # IDE_006 / TSK_004 — pin OpenMP / std::thread workers spawned by
    # the chosen kernel to the cores of this worker's local NUMA node
    # so split-K/V LSE-merge does not pay cross-socket UPI on its CPU
    # reads. Idempotent: cost is one ``os.sched_setaffinity`` call per
    # worker process. Silent no-op on single-socket dev / when the
    # platform has no NUMA topology.
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.numa_aware import (  # noqa: E501
            pin_threads_to_local_numa,
        )

        pin_threads_to_local_numa()
    except Exception:  # pragma: no cover - defence in depth
        pass

    path = _force_path if _force_path is not None else select_isa_path()

    common_kwargs = dict(
        query=query,
        cold_kv_cache=cold_kv_cache,
        cold_kv_layout=cold_kv_layout,
        cold_block_ids=cold_block_ids,
        cold_block_lens=cold_block_lens,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens_total=seq_lens_total,
        query_positions=query_positions,
        softmax_scale=softmax_scale,
        causal=causal,
        cold_kv_cache_v=cold_kv_cache_v,
    )

    # Split-K/V layout (FlashAttention's OffloadingConnector mirror) is
    # currently only supported on the Python reference path; the portable
    # C++ / AVX-512 / AMX kernels assume a single combined canonical
    # tensor with K and V back-to-back per page. Force PYTHON_REF when
    # split inputs are supplied so dispatch does not try to call a
    # kernel that ignores cold_kv_cache_v.
    if cold_kv_cache_v is not None:
        path = ISAPath.PYTHON_REF

    # AMX / AVX-512 are TSK_001 Phase 2 (prod). When ``_force_path``
    # picks one of them on a machine without the kernel built, fall
    # through to portable (or Python ref).
    if path == ISAPath.AMX:
        if _has_amx_kernel() and _has_amx():
            raise NotImplementedError(
                "AMX kernel binding not wired yet (TSK_001 §4.2b)"
            )
        # Cascade: AVX-512 → portable → Python ref
        path = (
            ISAPath.AVX512 if _has_avx512_kernel() and _has_avx512()
            else ISAPath.PORTABLE if _has_portable_kernel()
            else ISAPath.PYTHON_REF
        )

    if path == ISAPath.AVX512:
        if _has_avx512_kernel() and _has_avx512():
            raise NotImplementedError(
                "AVX-512 kernel binding not wired yet (TSK_001 §4.2a)"
            )
        path = (
            ISAPath.PORTABLE if _has_portable_kernel()
            else ISAPath.PYTHON_REF
        )

    if path == ISAPath.PORTABLE:
        if _has_portable_kernel():
            return _call_portable(**common_kwargs)
        # Portable kernel not loadable → fall through to Python ref.
        path = ISAPath.PYTHON_REF

    if path == ISAPath.PYTHON_REF:
        return python_reference_partial_attention(**common_kwargs)

    raise ValueError(f"unsupported ISA path: {path}")


def _call_portable(
    *,
    query: torch.Tensor,
    cold_kv_cache: torch.Tensor,
    cold_kv_layout: KVPageLayout,
    cold_block_ids: torch.Tensor,
    cold_block_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_total: torch.Tensor,
    query_positions: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    cold_kv_cache_v: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt Python-side inputs to the portable C++ kernel's signature.

    The C++ kernel expects:
      - all tensors on CPU
      - ``cold_block_ids``, ``cold_block_lens``, ``cu_seqlens_q``,
        ``query_positions`` as int32
      - ``cold_kv_cache`` as int8 (combined K+V, back-to-back per page)

    ``cold_kv_cache_v`` exists only for kwargs-unpacking compatibility
    with the dispatcher; the portable C++ kernel does not yet support
    split-K/V layouts so the dispatcher is expected to have forced
    PYTHON_REF before reaching this function.
    """
    if cold_kv_cache_v is not None:
        # Defence in depth — should never trigger because
        # forward_partial_with_lse forces PYTHON_REF in split mode.
        raise RuntimeError(
            "portable C++ kernel does not yet support split-K/V cold "
            "layout; forward_partial_with_lse should have routed this "
            "call to the Python reference."
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(cold_kv_layout.head_dim)

    def _i32(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int32) if t.dtype is not torch.int32 else t

    O, LSE = _PORTABLE_MOD.forward_partial_with_lse_portable(
        query.contiguous(),
        cold_kv_cache.contiguous(),
        int(cold_kv_layout.block_size),
        int(cold_kv_layout.num_kv_heads),
        int(cold_kv_layout.head_dim),
        int(cold_kv_layout.kv_block_bytes),
        _i32(cold_block_ids).contiguous(),
        _i32(cold_block_lens).contiguous(),
        _i32(cu_seqlens_q).contiguous(),
        _i32(query_positions).contiguous(),
        float(softmax_scale),
        bool(causal),
    )
    return O, LSE
