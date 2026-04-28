# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_010 — sub-batching numerical equivalence + boundary cases.

forward_partial_with_lse_sub_batched 의 결과가 single-batch 결과와 numerical
equivalence (BF16 ~5e-3 tolerance) 임을 확인. 또한 num_sub_batches=1 / num_seqs<=1
의 fallback 동작 검증.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.cpu_partial_attention import (
    forward_partial_with_lse,
    forward_partial_with_lse_sub_batched,
)
from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


def _build_simple_inputs(num_seqs: int, dtype: torch.dtype):
    """Small synthetic batch — Q×KV head 작게 + cold blocks 다수."""
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 32
    block_size = 8
    n_cold_blocks_per_seq = 3
    seq_len_per = n_cold_blocks_per_seq * block_size

    torch.manual_seed(0)

    # Layout + CPU page cache.
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=dtype,
    )
    num_cpu_blocks = num_seqs * n_cold_blocks_per_seq + 4  # extra for safety
    cpu_buf = torch.empty(num_cpu_blocks, layout.page_size_bytes, dtype=torch.int8)
    adapter = KVViewAdapter(cpu_buf, layout)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    with torch.no_grad():
        adapter.k_view().copy_(
            torch.randn(
                num_cpu_blocks, block_size, num_kv_heads, head_dim,
                generator=g1, dtype=dtype,
            )
        )
        adapter.v_view().copy_(
            torch.randn(
                num_cpu_blocks, block_size, num_kv_heads, head_dim,
                generator=g2, dtype=dtype,
            )
        )

    # Inputs.
    num_tokens = num_seqs  # decode-only: q_len=1 per seq
    query = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype)
    query_positions = torch.tensor(
        [seq_len_per - 1] * num_seqs, dtype=torch.int32
    )
    cu_seqlens_q = torch.arange(0, num_tokens + 1, dtype=torch.int32)
    seq_lens_total = torch.tensor([seq_len_per] * num_seqs, dtype=torch.int32)
    cold_block_ids = torch.zeros(num_seqs, n_cold_blocks_per_seq, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(n_cold_blocks_per_seq):
            cold_block_ids[s, b] = s * n_cold_blocks_per_seq + b
    cold_block_lens = torch.tensor(
        [n_cold_blocks_per_seq] * num_seqs, dtype=torch.int32
    )

    softmax_scale = 1.0 / (head_dim ** 0.5)

    return dict(
        query=query,
        cold_kv_cache=cpu_buf,
        cold_kv_layout=layout,
        cold_block_ids=cold_block_ids,
        cold_block_lens=cold_block_lens,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens_total=seq_lens_total,
        query_positions=query_positions,
        softmax_scale=softmax_scale,
        causal=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_sub_batches", [2, 3, 4])
def test_sub_batched_matches_single(dtype, num_sub_batches):
    """sub_batched 결과가 single-batch 결과와 BF16 tolerance 내 일치."""
    inputs = _build_simple_inputs(num_seqs=6, dtype=dtype)

    O_single, LSE_single = forward_partial_with_lse(**inputs)
    O_sub, LSE_sub = forward_partial_with_lse_sub_batched(
        **inputs, num_sub_batches=num_sub_batches
    )

    atol = 5e-3 if dtype is torch.bfloat16 else 1e-3
    rtol = 5e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(O_sub, O_single, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_sub, LSE_single, atol=atol, rtol=rtol)


def test_sub_batched_fallback_num_sub_batches_1():
    """num_sub_batches=1 → single-batch fallback (결과 동일)."""
    inputs = _build_simple_inputs(num_seqs=4, dtype=torch.bfloat16)

    O_single, LSE_single = forward_partial_with_lse(**inputs)
    O_sub, LSE_sub = forward_partial_with_lse_sub_batched(
        **inputs, num_sub_batches=1
    )

    torch.testing.assert_close(O_sub, O_single, atol=0, rtol=0)
    torch.testing.assert_close(LSE_sub, LSE_single, atol=0, rtol=0)


def test_sub_batched_fallback_single_seq():
    """num_seqs=1 → fallback (sub-batching 의미 없음)."""
    inputs = _build_simple_inputs(num_seqs=1, dtype=torch.bfloat16)

    O_single, LSE_single = forward_partial_with_lse(**inputs)
    O_sub, LSE_sub = forward_partial_with_lse_sub_batched(
        **inputs, num_sub_batches=4
    )

    torch.testing.assert_close(O_sub, O_single, atol=0, rtol=0)
    torch.testing.assert_close(LSE_sub, LSE_single, atol=0, rtol=0)


def test_sub_batched_more_groups_than_seqs():
    """num_sub_batches > num_seqs → 일부 group 이 empty, 결과 일치."""
    inputs = _build_simple_inputs(num_seqs=2, dtype=torch.bfloat16)

    O_single, LSE_single = forward_partial_with_lse(**inputs)
    O_sub, LSE_sub = forward_partial_with_lse_sub_batched(
        **inputs, num_sub_batches=8
    )

    atol = 5e-3
    rtol = 5e-2
    torch.testing.assert_close(O_sub, O_single, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_sub, LSE_single, atol=atol, rtol=rtol)
