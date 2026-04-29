# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO-style CPU paged attention — output-only interface.

Adapted from NEO ``pacpu/`` (MLSys 2025, Apache 2.0). Algorithms only.

This module exposes ``forward_attention`` — a CPU paged-attention
kernel front-end that returns only the attention output, without an
LSE side channel. NEO does not perform LSE-merging because each
request's KV is *exclusively* on one device (TSK_015 invariant), so
there is no GPU/CPU partial to merge.

A pure-Python reference implementation is provided for unit tests
and dev validation. The production AVX-512 / AMX kernels from the
IDE_006 hot/cold split branch will be cherry-picked into ``csrc/cpu/``
as part of TSK_018's later phases — at that point this module's
``forward_attention`` will dispatch to the SIMD kernel for the
``preferred_kernel`` argument.

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §7.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal


KernelKind = Literal["portable", "avx512", "amx"]


def forward_attention(
    *,
    q: list[list[list[float]]],          # [num_seqs, num_q_heads, head_dim]
    # Per-layer per-block KV cache:
    # [num_layers, num_blocks, num_kv_heads, block_size, head_dim].
    # Reference path uses Python lists; production SIMD kernels will
    # take a torch.Tensor view and zero-copy into a strided buffer.
    k_cache: list[list[list[list[list[float]]]]],
    v_cache: list[list[list[list[list[float]]]]],
    block_table: list[list[int]],         # [num_seqs, max_blocks_per_seq]
    seq_lens: list[int],                  # [num_seqs]
    cur_layer: int,
    softmax_scale: float,
    output: list[list[list[float]]],      # filled in-place
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    preferred_kernel: KernelKind = "portable",
) -> None:
    """Compute paged attention on the CPU. Writes result into ``output``.

    No LSE is returned — NEO's exclusive ownership invariant means the
    GPU never holds a partial that we'd need to merge with.

    The ``preferred_kernel`` argument selects between the dynamic
    dispatch tiers when the production kernels are wired in:
      * ``portable`` — pure C++ (always available, no SIMD)
      * ``avx512``   — AVX-512F (gated by cpuid, falls back to portable)
      * ``amx``      — AMX-BF16 (gated by cpuid, falls back to portable)

    Until the SIMD kernels are cherry-picked from the IDE_006 branch,
    only ``portable`` is functional and the wrapper logs a warning if
    a SIMD tier is requested.
    """
    if preferred_kernel != "portable":
        # Production cherry-pick wires AVX-512 / AMX paths here.
        preferred_kernel = "portable"

    qh_per_kvh = num_q_heads // num_kv_heads
    if qh_per_kvh * num_kv_heads != num_q_heads:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

    for seq_idx, seq_len in enumerate(seq_lens):
        if seq_len <= 0:
            continue
        seq_q = q[seq_idx]
        seq_o = output[seq_idx]

        # Per-Q-head accumulators.
        head_logits: list[list[float]] = [[] for _ in range(num_q_heads)]
        head_v_at_pos: list[list[list[float]]] = [
            [] for _ in range(num_q_heads)
        ]

        # Walk the seq_len tokens block by block.
        num_blocks_seq = (seq_len + block_size - 1) // block_size
        for blk_pos in range(num_blocks_seq):
            blk_id = block_table[seq_idx][blk_pos]
            tlim = min(block_size, seq_len - blk_pos * block_size)
            for kh in range(num_kv_heads):
                k_block = k_cache[cur_layer][blk_id][kh]   # [block_size, head_dim]
                v_block = v_cache[cur_layer][blk_id][kh]
                for slot in range(tlim):
                    k_vec = k_block[slot]
                    v_vec = v_block[slot]
                    for offs in range(qh_per_kvh):
                        qh = kh * qh_per_kvh + offs
                        # dot(q[qh], k_vec)
                        s = 0.0
                        qrow = seq_q[qh]
                        for d in range(head_dim):
                            s += qrow[d] * k_vec[d]
                        s *= softmax_scale
                        head_logits[qh].append(s)
                        head_v_at_pos[qh].append(v_vec)

        # Stable softmax + weighted sum per Q-head.
        for qh in range(num_q_heads):
            logits = head_logits[qh]
            if not logits:
                for d in range(head_dim):
                    seq_o[qh][d] = 0.0
                continue
            mx = max(logits)
            exps = [math.exp(s - mx) for s in logits]
            denom = sum(exps)
            inv = 1.0 / denom if denom > 0 else 0.0
            out = [0.0] * head_dim
            v_at = head_v_at_pos[qh]
            for w, v_vec in zip(exps, v_at):
                w *= inv
                for d in range(head_dim):
                    out[d] += w * v_vec[d]
            for d in range(head_dim):
                seq_o[qh][d] = out[d]
