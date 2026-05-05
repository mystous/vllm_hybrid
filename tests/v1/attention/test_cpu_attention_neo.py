# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/v1/attention/ops/cpu_attention.py`` (TSK_018).

Verifies the NEO-style output-only ``forward_attention`` reference
implementation. Production AVX-512 / AMX SIMD kernels will be wired
into the same interface in a later phase of TSK_018 (cherry-pick
from the IDE_006 hot/cold split branch).
"""

from __future__ import annotations

import math

import pytest

from vllm.v1.attention.ops.cpu_attention import forward_attention


# ----------------------------------------------------------------------
# Single-head sanity
# ----------------------------------------------------------------------
def test_forward_attention_softmax_weighted_sum_single_head():
    q = [[[1.0, 0.0, 0.0, 0.0]]]
    k_cache = [[[[[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]]]]]
    v_cache = [[[[[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]]]]]
    output = [[[0.0] * 4]]

    forward_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=[[0]], seq_lens=[2],
        cur_layer=0, softmax_scale=1.0, output=output,
        block_size=2, num_q_heads=1, num_kv_heads=1, head_dim=4,
    )

    # logits = [1.0, 0.0]; stable softmax → exp / sum
    e0, e1 = math.exp(0.0), math.exp(-1.0)
    s = e0 + e1
    w0, w1 = e0 / s, e1 / s
    expected = [w0 * 1.0 + w1 * 5.0,
                w0 * 2.0 + w1 * 6.0,
                w0 * 3.0 + w1 * 7.0,
                w0 * 4.0 + w1 * 8.0]
    for actual, exp in zip(output[0][0], expected):
        assert actual == pytest.approx(exp, rel=1e-6)


def test_forward_attention_zero_seq_skips():
    """seq_len=0 must leave output untouched and not raise."""
    q = [[[0.0]]]
    k_cache = [[[[[0.0]]]]]
    v_cache = [[[[[0.0]]]]]
    output = [[[1.5]]]   # sentinel
    forward_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=[[0]], seq_lens=[0],
        cur_layer=0, softmax_scale=1.0, output=output,
        block_size=1, num_q_heads=1, num_kv_heads=1, head_dim=1,
    )
    assert output == [[[1.5]]]


# ----------------------------------------------------------------------
# GQA — q_heads / kv_heads broadcast
# ----------------------------------------------------------------------
def test_forward_attention_gqa_broadcast_groups_q_heads():
    """q_heads=4 over kv_heads=2 (qh_per_kvh=2). Q-heads 0/1 share KV
    head 0; Q-heads 2/3 share KV head 1. The result of each Q-head is
    a softmax-weighted sum of the *same* V values for each KV-group."""
    q = [[
        [1.0, 0.0],   # qh 0
        [0.0, 1.0],   # qh 1 — same kv_head as qh 0
        [1.0, 0.0],   # qh 2
        [0.0, 1.0],   # qh 3 — same kv_head as qh 2
    ]]
    k_cache = [[[
        [[1.0, 0.0], [0.0, 1.0]],   # kv 0
        [[1.0, 0.0], [0.0, 1.0]],   # kv 1
    ]]]
    v_cache = [[[
        [[1.0, 1.0], [2.0, 2.0]],
        [[3.0, 3.0], [4.0, 4.0]],
    ]]]
    output = [[[0.0, 0.0] for _ in range(4)]]
    forward_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=[[0]], seq_lens=[2],
        cur_layer=0, softmax_scale=1.0, output=output,
        block_size=2, num_q_heads=4, num_kv_heads=2, head_dim=2,
    )
    # Q-head 0 and Q-head 2 use the same V vectors but different KV
    # heads; outputs should differ along the qh→kv mapping.
    assert output[0][0] != output[0][2]
    # Q-head 0 and Q-head 1 share KV head 0 — outputs differ because q
    # vectors differ.
    assert output[0][0] != output[0][1]


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_forward_attention_rejects_non_divisible_q_heads():
    with pytest.raises(ValueError, match="must be divisible"):
        forward_attention(
            q=[[[0.0]]], k_cache=[[[[[0.0]]]]], v_cache=[[[[[0.0]]]]],
            block_table=[[0]], seq_lens=[1],
            cur_layer=0, softmax_scale=1.0, output=[[[0.0]]],
            block_size=1, num_q_heads=3, num_kv_heads=2, head_dim=1,
        )


def test_forward_attention_simd_kernel_falls_back_to_portable():
    """Until TSK_018 cherry-picks the production SIMD kernels, asking
    for ``avx512`` or ``amx`` should silently fall back."""
    q = [[[1.0]]]
    k_cache = [[[[[1.0]]]]]
    v_cache = [[[[[2.0]]]]]
    out = [[[0.0]]]
    forward_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=[[0]], seq_lens=[1],
        cur_layer=0, softmax_scale=1.0, output=out,
        block_size=1, num_q_heads=1, num_kv_heads=1, head_dim=1,
        preferred_kernel="avx512",
    )
    assert out[0][0][0] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# Multi-block
# ----------------------------------------------------------------------
def test_forward_attention_spans_multiple_blocks():
    """seq_len exceeds block_size, so the kernel walks more than one
    block in the block table."""
    block_size = 2
    seq_len = 3      # 2 blocks (2 + 1 tokens)
    head_dim = 1
    q = [[[1.0]]]
    k_cache = [[
        [[[1.0], [1.0]]],
        [[[1.0], [0.0]]],
    ]]
    v_cache = [[
        [[[1.0], [2.0]]],
        [[[3.0], [0.0]]],
    ]]
    out = [[[0.0]]]
    forward_attention(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=[[0, 1]], seq_lens=[seq_len],
        cur_layer=0, softmax_scale=1.0, output=out,
        block_size=block_size, num_q_heads=1, num_kv_heads=1,
        head_dim=head_dim,
    )
    # Three logits: 1, 1, 1; uniform softmax → mean of [1, 2, 3] = 2.0
    assert out[0][0][0] == pytest.approx(2.0)
