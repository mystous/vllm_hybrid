# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TST_004 — TSK_003 §4.2b AMX kernel cross-check (B(iii)).

Verifies the AMX BF16 kernel agrees with the portable C++ reference
within BF16 round-off tolerance. Skipped on hosts that do not expose
AMX (most dev boxes — Sapphire Rapids and newer Xeon parts only).

The chain ``Python ref → portable C++ → AMX C++`` is closed via
``test_portable_cross_check.py`` (Python ref vs portable) and this
file (portable vs AMX). FP16 / FP32 dispatch falls back to the AMX
translation unit's AVX-512 path internally; we still cross-check it
so any divergence between the AMX kernel's FP16/FP32 fallback and the
portable kernel is caught.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.ops.cpu_partial_attention import (
    ISAPath,
    forward_partial_with_lse,
    _has_amx_kernel,
    _has_portable_kernel,
)
from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


pytestmark = pytest.mark.skipif(
    not (_has_amx_kernel() and _has_portable_kernel()),
    reason=(
        "AMX cross-check skipped — needs both portable and AMX C++ "
        "kernels buildable, plus host CPU exposing AMX-BF16. AMX is a "
        "Sapphire Rapids+ feature (Xeon 8xxx), so consumer / older "
        "server CPUs auto-skip. The test re-activates automatically on "
        "any host where ``_has_amx_kernel()`` returns True."
    ),
)


def _make_inputs(
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
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=kv_dtype,
    )
    canonical = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.int8
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
        torch.randn(
            q_len, num_q_heads, head_dim, generator=gen, dtype=torch.float32
        ).to(kv_dtype)
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
            n_cold, n_cold + q_len, dtype=torch.int32
        ),
        softmax_scale=1.0 / math.sqrt(head_dim),
    )


@pytest.fixture(params=[torch.bfloat16, torch.float16])
def kv_dtype(request):
    return request.param


@pytest.fixture(params=[64, 128])
def head_dim(request):
    return request.param


@pytest.fixture
def num_kv_heads():
    return 4


@pytest.fixture
def block_size():
    # AMX kernel batches 16 K rows per tile call. block_size must be a
    # multiple of 16 to keep the per-batch K rows in the same canonical
    # block (PLN_001 §3 scope: block_size ∈ {16, 32, 64} all qualify).
    return 16


@pytest.mark.parametrize("num_blocks", [1, 4, 8])
@pytest.mark.parametrize("q_len", [1, 4, 16])
def test_amx_matches_portable_single_seq(
    kv_dtype, head_dim, num_kv_heads, block_size, num_blocks, q_len
):
    inputs = _make_inputs(
        kv_dtype=kv_dtype,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_blocks=num_blocks,
        block_size=block_size,
        q_len=q_len,
        num_q_heads=num_kv_heads * 4,
        seed=42,
    )

    O_por, LSE_por = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE
    )
    O_amx, LSE_amx = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.AMX
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_amx, O_por, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_amx, LSE_por, atol=1e-3, rtol=1e-3)


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
    """Same as ``_make_inputs`` but with K and V in two separate
    canonical int8 buffers (FlashAttention's OffloadingConnector mirror
    layout — the prod path that actually drives this kernel)."""
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=kv_dtype,
    )
    k_canonical = torch.zeros((num_blocks, layout.kv_block_bytes), dtype=torch.int8)
    v_canonical = torch.zeros((num_blocks, layout.kv_block_bytes), dtype=torch.int8)
    adapter = KVViewAdapter.from_split_kv(k_canonical, v_canonical, layout)
    K = adapter.k_view()
    V = adapter.v_view()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    K.copy_(torch.randn(K.shape, generator=gen, dtype=torch.float32).to(kv_dtype) * 0.5)
    V.copy_(torch.randn(V.shape, generator=gen, dtype=torch.float32).to(kv_dtype) * 0.5)
    n_cold = num_blocks * block_size
    Q = (
        torch.randn(q_len, num_q_heads, head_dim, generator=gen, dtype=torch.float32)
        .to(kv_dtype) * 0.5
    )
    return dict(
        query=Q,
        cold_kv_cache=k_canonical,
        cold_kv_cache_v=v_canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor([list(range(num_blocks))], dtype=torch.int32),
        cold_block_lens=torch.tensor([num_blocks], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens_total=torch.tensor([n_cold + q_len], dtype=torch.int32),
        query_positions=torch.arange(n_cold, n_cold + q_len, dtype=torch.int32),
        softmax_scale=1.0 / math.sqrt(head_dim),
    )


@pytest.mark.parametrize("num_blocks", [1, 4, 8])
@pytest.mark.parametrize("q_len", [1, 4, 16])
def test_amx_split_kv_matches_portable(
    kv_dtype, head_dim, num_kv_heads, block_size, num_blocks, q_len
):
    """Split-K/V layout cross-check — exercises the same code path
    that prod uses (OffloadingConnector mirror has K and V in two
    separate buffers)."""
    inputs = _make_split_inputs(
        kv_dtype=kv_dtype,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_blocks=num_blocks,
        block_size=block_size,
        q_len=q_len,
        num_q_heads=num_kv_heads * 4,
        seed=43,
    )

    O_por, LSE_por = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE
    )
    O_amx, LSE_amx = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.AMX
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_amx, O_por, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_amx, LSE_por, atol=1e-3, rtol=1e-3)


def test_amx_matches_portable_multi_seq(kv_dtype, head_dim, num_kv_heads):
    """Multi-sequence batch with variable cold-block lengths — exercises
    cross-sequence pointer arithmetic and AMX tile reconfig."""
    block_size = 16
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=kv_dtype,
    )
    num_blocks = 6
    canonical = torch.zeros(
        (num_blocks, layout.page_size_bytes), dtype=torch.int8
    )
    adapter = KVViewAdapter(canonical, layout)
    K, V = adapter.k_view(), adapter.v_view()
    gen = torch.Generator(device="cpu").manual_seed(7)
    K.copy_(
        torch.randn(K.shape, generator=gen, dtype=torch.float32).to(kv_dtype)
        * 0.5
    )
    V.copy_(
        torch.randn(V.shape, generator=gen, dtype=torch.float32).to(kv_dtype)
        * 0.5
    )

    num_q_heads = num_kv_heads * 2
    cu_seqlens_q = torch.tensor([0, 2, 6], dtype=torch.int32)
    cold_block_ids = torch.tensor(
        [[0, 1, 2], [4, 5, -1]], dtype=torch.int32
    )
    cold_block_lens = torch.tensor([3, 2], dtype=torch.int32)
    n_cold0 = 3 * block_size
    n_cold1 = 2 * block_size
    seq_lens_total = torch.tensor(
        [n_cold0 + 2, n_cold1 + 4], dtype=torch.int32
    )
    query_positions = torch.tensor(
        [
            n_cold0, n_cold0 + 1,
            n_cold1, n_cold1 + 1, n_cold1 + 2, n_cold1 + 3,
        ],
        dtype=torch.int32,
    )
    Q = (
        torch.randn(6, num_q_heads, head_dim, generator=gen, dtype=torch.float32)
        .to(kv_dtype) * 0.5
    )
    inputs = dict(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=cold_block_ids,
        cold_block_lens=cold_block_lens,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens_total=seq_lens_total,
        query_positions=query_positions,
        softmax_scale=1.0 / math.sqrt(head_dim),
    )

    O_por, LSE_por = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE
    )
    O_amx, LSE_amx = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.AMX
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_amx, O_por, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_amx, LSE_por, atol=1e-3, rtol=1e-3)
