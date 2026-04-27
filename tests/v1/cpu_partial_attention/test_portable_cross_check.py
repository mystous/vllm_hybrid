# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for TSK_001 §4.2c portable C++ kernel — TST_001 단계 B(i).

The portable C++ kernel is JIT-compiled on first import (see
``vllm/v1/attention/ops/cpu_partial_attention.py``). We verify it
produces results numerically consistent with the Python reference
across the sweep dimensions PLN_001 §4.1 calls out.

Tolerance: BF16 / FP16 partial attention has limited intermediate
precision; the C++ scalar accumulator and Python torch ops differ in
how they arrange the dot product, so we accept BF16 ~5e-3 / FP16
~1e-3 absolute error.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.ops.cpu_partial_attention import (
    ISAPath,
    forward_partial_with_lse,
    python_reference_partial_attention,
    _has_portable_kernel,
)
from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


pytestmark = pytest.mark.skipif(
    not _has_portable_kernel(),
    reason="portable C++ kernel not buildable in this environment",
)


def _make_inputs(
    *, kv_dtype, head_dim, num_kv_heads, num_blocks, block_size,
    q_len, num_q_heads, seed=0,
):
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    canonical = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.int8,
    )
    adapter = KVViewAdapter(canonical, layout)
    K = adapter.k_view()
    V = adapter.v_view()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    K.copy_(
        torch.randn(K.shape, generator=gen, dtype=torch.float32).to(kv_dtype)
        * 0.5
    )
    V.copy_(
        torch.randn(V.shape, generator=gen, dtype=torch.float32).to(kv_dtype)
        * 0.5
    )

    n_cold = num_blocks * block_size
    Q = (
        torch.randn(q_len, num_q_heads, head_dim, generator=gen,
                    dtype=torch.float32).to(kv_dtype)
        * 0.5
    )
    return dict(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor(
            [list(range(num_blocks))], dtype=torch.int32
        ),
        cold_block_lens=torch.tensor([num_blocks], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens_total=torch.tensor([n_cold + q_len], dtype=torch.int32),
        query_positions=torch.arange(
            n_cold, n_cold + q_len, dtype=torch.int32,
        ),
        softmax_scale=1.0 / math.sqrt(head_dim),
    )


# ---------------------------------------------------------------------
# B(i) Python reference vs portable C++ (single sequence, sweep)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("num_blocks", [1, 4, 8])
@pytest.mark.parametrize("q_len", [1, 4, 16])
def test_portable_matches_python_ref_single_seq(
    kv_dtype, head_dim, num_kv_heads, block_size, num_blocks, q_len,
):
    inputs = _make_inputs(
        kv_dtype=kv_dtype, head_dim=head_dim,
        num_kv_heads=num_kv_heads, num_blocks=num_blocks,
        block_size=block_size, q_len=q_len,
        num_q_heads=num_kv_heads * 4,  # GQA q_per_kv=4
        seed=42,
    )

    O_ref, LSE_ref = python_reference_partial_attention(**inputs)
    O_por, LSE_por = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE,
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_por, O_ref, atol=atol, rtol=rtol)
    # LSE in fp32 — tighter tolerance
    torch.testing.assert_close(LSE_por, LSE_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------
# B(i) — multi-sequence, variable cold-block lengths
# ---------------------------------------------------------------------


def test_portable_matches_python_ref_multi_seq(
    kv_dtype, head_dim, num_kv_heads,
):
    block_size = 16
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    num_blocks = 6
    canonical = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.int8,
    )
    adapter = KVViewAdapter(canonical, layout)
    K, V = adapter.k_view(), adapter.v_view()
    gen = torch.Generator(device="cpu").manual_seed(7)
    K.copy_(torch.randn(K.shape, generator=gen, dtype=torch.float32).to(kv_dtype) * 0.5)
    V.copy_(torch.randn(V.shape, generator=gen, dtype=torch.float32).to(kv_dtype) * 0.5)

    num_q_heads = num_kv_heads * 2
    # Seq 0: 3 cold blocks [0, 1, 2], q_len=2
    # Seq 1: 2 cold blocks [4, 5],    q_len=4
    cu_seqlens_q = torch.tensor([0, 2, 6], dtype=torch.int32)
    cold_block_ids = torch.tensor(
        [[0, 1, 2], [4, 5, -1]], dtype=torch.int32,
    )
    cold_block_lens = torch.tensor([3, 2], dtype=torch.int32)
    n_cold0 = 3 * block_size
    n_cold1 = 2 * block_size
    seq_lens_total = torch.tensor(
        [n_cold0 + 2, n_cold1 + 4], dtype=torch.int32,
    )
    query_positions = torch.tensor(
        [n_cold0, n_cold0 + 1,
         n_cold1, n_cold1 + 1, n_cold1 + 2, n_cold1 + 3],
        dtype=torch.int32,
    )

    Q = torch.randn(6, num_q_heads, head_dim, generator=gen,
                    dtype=torch.float32).to(kv_dtype) * 0.5

    common = dict(
        query=Q, cold_kv_cache=canonical, cold_kv_layout=layout,
        cold_block_ids=cold_block_ids, cold_block_lens=cold_block_lens,
        cu_seqlens_q=cu_seqlens_q, seq_lens_total=seq_lens_total,
        query_positions=query_positions,
        softmax_scale=1.0 / math.sqrt(head_dim),
    )
    O_ref, LSE_ref = python_reference_partial_attention(**common)
    O_por, LSE_por = forward_partial_with_lse(**common, _force_path=ISAPath.PORTABLE)

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_por, O_ref, atol=atol, rtol=atol)
    torch.testing.assert_close(LSE_por, LSE_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------
# B(i) — causal mask: portable vs python ref must agree on masking
# ---------------------------------------------------------------------


def test_portable_causal_mask_matches_python_ref(
    kv_dtype, head_dim, num_kv_heads,
):
    block_size = 16
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    num_blocks = 1
    canonical = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.int8,
    )
    adapter = KVViewAdapter(canonical, layout)
    gen = torch.Generator(device="cpu").manual_seed(11)
    adapter.k_view().copy_(
        torch.randn(adapter.k_view().shape, generator=gen,
                    dtype=torch.float32).to(kv_dtype) * 0.5
    )
    adapter.v_view().copy_(
        torch.randn(adapter.v_view().shape, generator=gen,
                    dtype=torch.float32).to(kv_dtype) * 0.5
    )

    num_q_heads = num_kv_heads
    Q = torch.randn(1, num_q_heads, head_dim, generator=gen,
                    dtype=torch.float32).to(kv_dtype) * 0.5
    common = dict(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor([[0]], dtype=torch.int32),
        cold_block_lens=torch.tensor([1], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_total=torch.tensor([16], dtype=torch.int32),
        query_positions=torch.tensor([5], dtype=torch.int32),  # mid-block
        softmax_scale=1.0 / math.sqrt(head_dim),
    )

    O_ref, LSE_ref = python_reference_partial_attention(**common)
    O_por, LSE_por = forward_partial_with_lse(**common, _force_path=ISAPath.PORTABLE)

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_por, O_ref, atol=atol, rtol=atol)
    torch.testing.assert_close(LSE_por, LSE_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------
# Long-context simulation (TST_001 §3.1 ctx_len=8192 cell — dev sanity)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("ctx_len", [2048, 8192])
def test_portable_long_context_sanity(
    kv_dtype, head_dim, num_kv_heads, ctx_len,
):
    block_size = 16
    num_blocks = ctx_len // block_size
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    canonical = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.int8,
    )
    adapter = KVViewAdapter(canonical, layout)
    gen = torch.Generator(device="cpu").manual_seed(99)
    adapter.k_view().copy_(
        torch.randn(adapter.k_view().shape, generator=gen,
                    dtype=torch.float32).to(kv_dtype) * 0.5
    )
    adapter.v_view().copy_(
        torch.randn(adapter.v_view().shape, generator=gen,
                    dtype=torch.float32).to(kv_dtype) * 0.5
    )
    num_q_heads = num_kv_heads * 4
    q_len = 2  # decode-style: very few new tokens per step

    common = dict(
        query=(
            torch.randn(q_len, num_q_heads, head_dim, generator=gen,
                        dtype=torch.float32).to(kv_dtype) * 0.5
        ),
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor(
            [list(range(num_blocks))], dtype=torch.int32,
        ),
        cold_block_lens=torch.tensor([num_blocks], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens_total=torch.tensor([ctx_len + q_len], dtype=torch.int32),
        query_positions=torch.tensor(
            [ctx_len, ctx_len + 1], dtype=torch.int32,
        ),
        softmax_scale=1.0 / math.sqrt(head_dim),
    )

    O_ref, LSE_ref = python_reference_partial_attention(**common)
    O_por, LSE_por = forward_partial_with_lse(**common, _force_path=ISAPath.PORTABLE)

    # BF16 with very long context accumulates more error — bump atol.
    atol = 1e-2 if kv_dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(O_por, O_ref, atol=atol, rtol=atol)
    torch.testing.assert_close(LSE_por, LSE_ref, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------
# Default dispatch should pick PORTABLE on this dev machine
# (assuming portable kernel built and AMX/AVX-512 not wired yet).
# ---------------------------------------------------------------------


def test_default_dispatch_uses_portable_when_available(
    kv_dtype, head_dim, num_kv_heads,
):
    inputs = _make_inputs(
        kv_dtype=kv_dtype, head_dim=head_dim,
        num_kv_heads=num_kv_heads, num_blocks=2, block_size=16,
        q_len=2, num_q_heads=num_kv_heads * 2, seed=3,
    )

    # Default path
    O_default, LSE_default = forward_partial_with_lse(**inputs)
    # Force portable
    O_force, LSE_force = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE,
    )
    torch.testing.assert_close(O_default, O_force, atol=0, rtol=0)
    torch.testing.assert_close(LSE_default, LSE_force, atol=0, rtol=0)


# ---------------------------------------------------------------------
# Split-K/V layout — FlashAttention's OffloadingConnector mirror passes
# K and V as two separate canonical int8 buffers. The portable kernel
# reads them via the (cold_kv_cache, cold_kv_cache_v) parameter pair.
# ---------------------------------------------------------------------


def _make_split_inputs(
    *,
    kv_dtype,
    head_dim: int,
    num_kv_heads: int,
    num_blocks: int,
    block_size: int,
    q_len: int,
    num_q_heads: int,
    seed: int = 0,
):
    """Build an input dict using two separate K-only and V-only int8
    canonical buffers (split-K/V layout). The returned dict is shaped
    for ``forward_partial_with_lse``: ``cold_kv_cache`` carries the K
    buffer and ``cold_kv_cache_v`` carries the V buffer.
    """
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=kv_dtype,
    )
    k_canonical = torch.zeros(
        (num_blocks, layout.kv_block_bytes), dtype=torch.int8
    )
    v_canonical = torch.zeros(
        (num_blocks, layout.kv_block_bytes), dtype=torch.int8
    )
    adapter = KVViewAdapter.from_split_kv(k_canonical, v_canonical, layout)
    K = adapter.k_view()
    V = adapter.v_view()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    K.copy_(
        torch.randn(K.shape, generator=gen, dtype=torch.float32).to(kv_dtype)
        * 0.5
    )
    V.copy_(
        torch.randn(V.shape, generator=gen, dtype=torch.float32).to(kv_dtype)
        * 0.5
    )
    n_cold = num_blocks * block_size
    Q = (
        torch.randn(
            q_len, num_q_heads, head_dim, generator=gen, dtype=torch.float32
        ).to(kv_dtype)
        * 0.5
    )
    return dict(
        query=Q,
        cold_kv_cache=k_canonical,
        cold_kv_cache_v=v_canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor(
            [list(range(num_blocks))], dtype=torch.int32
        ),
        cold_block_lens=torch.tensor([num_blocks], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens_total=torch.tensor([n_cold + q_len], dtype=torch.int32),
        query_positions=torch.arange(
            n_cold, n_cold + q_len, dtype=torch.int32
        ),
        softmax_scale=1.0 / math.sqrt(head_dim),
    )


@pytest.mark.parametrize("num_blocks", [1, 4, 8])
@pytest.mark.parametrize("q_len", [1, 4, 16])
def test_portable_split_kv_matches_python_ref(
    kv_dtype, head_dim, num_kv_heads, block_size, num_blocks, q_len,
):
    inputs = _make_split_inputs(
        kv_dtype=kv_dtype, head_dim=head_dim,
        num_kv_heads=num_kv_heads, num_blocks=num_blocks,
        block_size=block_size, q_len=q_len,
        num_q_heads=num_kv_heads * 4, seed=42,
    )

    O_ref, LSE_ref = python_reference_partial_attention(**inputs)
    O_por, LSE_por = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE,
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_por, O_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_por, LSE_ref, atol=1e-3, rtol=1e-3)
