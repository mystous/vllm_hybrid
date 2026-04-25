# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the Python reference partial attention impl —
TST_001 단계 B (i) baseline (Python reference vs naive ground truth).

The portable / AVX-512 / AMX C++ kernels are not built yet, so the
pair-wise comparisons (B(ii), B(iii), portable vs Python ref) are
deferred to a later round. For now we verify that the Python reference
itself matches a naive ground-truth implementation, which is the
"reference" that all SIMD paths will later be checked against.

See ``shadow_assists/features/IDE_006/TST_001.md`` §3.1·§4.1·§6.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.ops.cpu_partial_attention import (
    python_reference_partial_attention,
)
from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


# ---------------------------------------------------------------------
# Helpers — naive ground-truth attention + fixture builders
# ---------------------------------------------------------------------


def _naive_partial_attention(
    *,
    Q: torch.Tensor,                # [q_len, num_q_heads, head_dim]
    K: torch.Tensor,                # [kv_len, num_q_heads, head_dim]
    V: torch.Tensor,                # [kv_len, num_q_heads, head_dim]
    q_positions: torch.Tensor,      # [q_len]
    softmax_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive (slow but obviously correct) partial attention.

    Returns ``(O, LSE)`` — same shape conventions as the wrapper.
    """
    q_len, num_q_heads, head_dim = Q.shape
    kv_len = K.shape[0]
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()

    O = torch.zeros_like(Q)
    LSE = torch.full(
        (num_q_heads, q_len), float("-inf"), dtype=torch.float32,
    )

    if kv_len == 0:
        return O, LSE

    kv_positions = torch.arange(kv_len)

    for t in range(q_len):
        for h in range(num_q_heads):
            scores = (Qf[t, h] @ Kf[:, h].T) * softmax_scale  # [kv_len]
            if causal:
                mask = q_positions[t] >= kv_positions
                scores = scores.masked_fill(~mask, float("-inf"))
            if torch.isinf(scores).all():
                continue
            m = scores.max()
            ex = (scores - m).exp()
            sx = ex.sum()
            probs = ex / sx
            O[t, h] = (probs.unsqueeze(-1) * Vf[:, h]).sum(dim=0).to(Q.dtype)
            LSE[h, t] = (m + sx.log()).item()

    return O, LSE


def _make_synthetic_cold_kv(
    *,
    num_blocks: int,
    layout: KVPageLayout,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate canonical int8 cold KV cache and return matching K/V views.

    Returns ``(canonical, K, V)`` — K and V share storage with canonical
    via :class:`KVViewAdapter`. We initialise the typed views with
    deterministic random values then read back through the adapter.
    """
    canonical = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.int8,
    )
    adapter = KVViewAdapter(canonical, layout)
    K = adapter.k_view()
    V = adapter.v_view()

    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Bound values for numerical stability of softmax in BF16/FP16.
    K.copy_(torch.randn(K.shape, generator=gen, dtype=torch.float32).to(K.dtype) * 0.5)
    V.copy_(torch.randn(V.shape, generator=gen, dtype=torch.float32).to(V.dtype) * 0.5)

    return canonical, K, V


# ---------------------------------------------------------------------
# Sanity — empty cold blocks, single sequence, single query token
# ---------------------------------------------------------------------


def test_empty_cold_blocks_returns_zero_and_neg_inf_lse(
    kv_dtype, head_dim, num_kv_heads
):
    """No cold blocks → O = 0, LSE = -inf for all tokens."""
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=16, dtype=kv_dtype,
    )
    canonical, _, _ = _make_synthetic_cold_kv(
        num_blocks=0, layout=layout, seed=0,
    )
    num_q_heads = num_kv_heads * 2  # GQA q_per_kv=2
    num_tokens = 3

    Q = torch.randn(num_tokens, num_q_heads, head_dim, dtype=kv_dtype)
    O, LSE = python_reference_partial_attention(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.zeros((1, 1), dtype=torch.int32),
        cold_block_lens=torch.zeros((1,), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32),
        seq_lens_total=torch.tensor([num_tokens], dtype=torch.int32),
        query_positions=torch.arange(num_tokens, dtype=torch.int32),
    )

    assert torch.equal(O, torch.zeros_like(O))
    assert torch.isinf(LSE).all()
    assert (LSE < 0).all()


# ---------------------------------------------------------------------
# Reference vs naive ground-truth — same single sequence
# ---------------------------------------------------------------------


@pytest.mark.parametrize("q_len", [1, 4, 8])
def test_reference_matches_naive_single_sequence(
    kv_dtype, head_dim, num_kv_heads, q_len
):
    """For one sequence with full cold-KV history, reference impl
    must match a naive O(q_len * kv_len) ground-truth."""
    block_size = 16
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    num_blocks = 2  # 2 blocks * 16 tokens = 32 cold KV tokens
    canonical, K_blocks, V_blocks = _make_synthetic_cold_kv(
        num_blocks=num_blocks, layout=layout, seed=42,
    )

    num_q_heads = num_kv_heads * 4  # Qwen2.5-7B style: q_per_kv=4 (smaller mock)
    q_per_kv = num_q_heads // num_kv_heads

    # Query positions are placed AFTER all cold KV tokens so the causal
    # mask doesn't drop anything.
    n_cold_kv_tokens = num_blocks * block_size  # 32
    query_positions = torch.arange(
        n_cold_kv_tokens, n_cold_kv_tokens + q_len, dtype=torch.int32,
    )
    seq_total = n_cold_kv_tokens + q_len

    gen = torch.Generator(device="cpu").manual_seed(7)
    Q = torch.randn(
        q_len, num_q_heads, head_dim, generator=gen, dtype=torch.float32,
    ).to(kv_dtype) * 0.5

    softmax_scale = 1.0 / math.sqrt(head_dim)

    O_ref, LSE_ref = python_reference_partial_attention(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor(
            [[i for i in range(num_blocks)]], dtype=torch.int32,
        ),
        cold_block_lens=torch.tensor([num_blocks], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens_total=torch.tensor([seq_total], dtype=torch.int32),
        query_positions=query_positions,
        softmax_scale=softmax_scale,
    )

    # Build naive K / V: flatten num_blocks × block_size, broadcast KV heads
    # to query heads.
    K_flat = K_blocks.reshape(-1, num_kv_heads, head_dim)
    V_flat = V_blocks.reshape(-1, num_kv_heads, head_dim)
    K_naive = K_flat.repeat_interleave(q_per_kv, dim=1)
    V_naive = V_flat.repeat_interleave(q_per_kv, dim=1)

    O_naive, LSE_naive = _naive_partial_attention(
        Q=Q, K=K_naive, V=V_naive,
        q_positions=query_positions,
        softmax_scale=softmax_scale,
    )

    # tolerance: BF16 path uses fp32 accumulate but final cast is in
    # query dtype. Allow a small relative error.
    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_ref, O_naive, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_ref, LSE_naive, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------
# Causal masking — query position < cold KV position must be masked out
# ---------------------------------------------------------------------


def test_causal_mask_drops_future_cold_kv(
    kv_dtype, head_dim, num_kv_heads
):
    """If a query is at position 5 and there are 10 cold KV tokens
    spanning positions 0..9, only positions 0..5 contribute."""
    block_size = 16
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    num_blocks = 1  # 16 cold KV positions
    canonical, K_blocks, V_blocks = _make_synthetic_cold_kv(
        num_blocks=num_blocks, layout=layout, seed=11,
    )
    num_q_heads = num_kv_heads
    q_per_kv = 1

    # Single query at absolute position 5 — only cold positions 0..5
    # are visible; positions 6..15 are masked.
    q_position = 5
    Q = torch.randn(1, num_q_heads, head_dim, dtype=kv_dtype) * 0.5

    O_ref, LSE_ref = python_reference_partial_attention(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor([[0]], dtype=torch.int32),
        cold_block_lens=torch.tensor([1], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_total=torch.tensor([16], dtype=torch.int32),
        query_positions=torch.tensor([q_position], dtype=torch.int32),
    )

    # Compare against naive on the visible prefix only.
    K_visible = K_blocks[0, : q_position + 1].repeat_interleave(q_per_kv, dim=1)
    V_visible = V_blocks[0, : q_position + 1].repeat_interleave(q_per_kv, dim=1)

    O_naive, LSE_naive = _naive_partial_attention(
        Q=Q, K=K_visible, V=V_visible,
        q_positions=torch.tensor([0], dtype=torch.int32),  # within visible
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=False,  # already truncated to visible KV
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_ref, O_naive, atol=atol, rtol=atol)


# ---------------------------------------------------------------------
# Multi-sequence (batch>1)
# ---------------------------------------------------------------------


def test_multi_sequence_independent_segments(kv_dtype, head_dim, num_kv_heads):
    """Two sequences with different cold-block lengths must be
    computed independently and produce results matching per-sequence
    naive computation."""
    block_size = 16
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    num_blocks = 4
    canonical, K_blocks, V_blocks = _make_synthetic_cold_kv(
        num_blocks=num_blocks, layout=layout, seed=99,
    )
    num_q_heads = num_kv_heads * 2

    # Seq 0: 2 cold blocks [block_id 0, 1], q_len=2
    # Seq 1: 1 cold block  [block_id 3],    q_len=3
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cold_block_ids = torch.tensor(
        [[0, 1, -1], [3, -1, -1]], dtype=torch.int32,
    )
    cold_block_lens = torch.tensor([2, 1], dtype=torch.int32)

    n_cold_seq0 = 2 * block_size  # 32
    n_cold_seq1 = 1 * block_size  # 16
    seq_lens_total = torch.tensor(
        [n_cold_seq0 + 2, n_cold_seq1 + 3], dtype=torch.int32,
    )
    # Place queries after their cold KV (no causal mask drops).
    query_positions = torch.tensor(
        [n_cold_seq0, n_cold_seq0 + 1,
         n_cold_seq1, n_cold_seq1 + 1, n_cold_seq1 + 2],
        dtype=torch.int32,
    )

    Q = torch.randn(5, num_q_heads, head_dim, dtype=kv_dtype) * 0.5

    O_ref, LSE_ref = python_reference_partial_attention(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=cold_block_ids,
        cold_block_lens=cold_block_lens,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens_total=seq_lens_total,
        query_positions=query_positions,
    )

    softmax_scale = 1.0 / math.sqrt(head_dim)
    q_per_kv = num_q_heads // num_kv_heads

    for s, (q_start, q_end, block_ids, n_cold_blocks) in enumerate(
        [(0, 2, [0, 1], 2), (2, 5, [3], 1)]
    ):
        block_id_t = torch.tensor(block_ids, dtype=torch.long)
        K_s = K_blocks.index_select(0, block_id_t).reshape(
            -1, num_kv_heads, head_dim
        ).repeat_interleave(q_per_kv, dim=1)
        V_s = V_blocks.index_select(0, block_id_t).reshape(
            -1, num_kv_heads, head_dim
        ).repeat_interleave(q_per_kv, dim=1)
        Q_s = Q[q_start:q_end]
        # Convert absolute query positions to KV-relative for naive.
        q_pos_rel = query_positions[q_start:q_end] - 0  # cold starts at 0
        O_naive_s, _ = _naive_partial_attention(
            Q=Q_s, K=K_s, V=V_s,
            q_positions=q_pos_rel.to(torch.long),
            softmax_scale=softmax_scale,
        )
        atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
        torch.testing.assert_close(
            O_ref[q_start:q_end], O_naive_s, atol=atol, rtol=atol,
        )
