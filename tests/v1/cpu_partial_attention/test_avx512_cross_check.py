# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TST_004 — TSK_003 §4.2a AVX-512 kernel cross-check (B(ii)).

Verifies the AVX-512 kernel agrees with the portable C++ reference
within BF16 / FP16 round-off tolerance. The portable kernel itself is
already validated against the Python reference in
``test_portable_cross_check.py``, so passing the AVX-512 kernel against
portable closes the chain ``Python ref → portable C++ → AVX-512 C++``.

Skipped when:
  * the host CPU does not expose AVX-512 (e.g. a fused-off Alder Lake
    where BIOS / microcode disabled the extension), OR
  * the AVX-512 kernel cannot be built (compiler too old, missing
    flags). Both cases collapse onto ``_has_avx512_kernel() == False``.

Tolerance mirrors ``test_portable_cross_check.py``: BF16 ~5e-3, FP16
~1e-3, plus 1e-3 for fp32 LSE.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.ops.cpu_partial_attention import (
    ISAPath,
    forward_partial_with_lse,
    _has_avx512_kernel,
    _has_portable_kernel,
)
from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


pytestmark = pytest.mark.skipif(
    not (_has_avx512_kernel() and _has_portable_kernel()),
    reason=(
        "AVX-512 cross-check skipped — needs both portable and AVX-512 "
        "C++ kernels buildable on this CPU. On a fused-off Alder Lake "
        "(microcode disabled AVX-512 even though the silicon has it) "
        "this skip is expected; the test re-activates automatically on "
        "a Sapphire Rapids+ host."
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
    return 16


@pytest.mark.parametrize("num_blocks", [1, 4, 8])
@pytest.mark.parametrize("q_len", [1, 4, 16])
def test_avx512_matches_portable_single_seq(
    kv_dtype, head_dim, num_kv_heads, block_size, num_blocks, q_len
):
    inputs = _make_inputs(
        kv_dtype=kv_dtype,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_blocks=num_blocks,
        block_size=block_size,
        q_len=q_len,
        num_q_heads=num_kv_heads * 4,  # GQA q_per_kv=4
        seed=42,
    )

    O_por, LSE_por = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE
    )
    O_avx, LSE_avx = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.AVX512
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_avx, O_por, atol=atol, rtol=rtol)
    # LSE in fp32 — tighter tolerance.
    torch.testing.assert_close(LSE_avx, LSE_por, atol=1e-3, rtol=1e-3)


def test_avx512_matches_portable_multi_seq(kv_dtype, head_dim, num_kv_heads):
    """Two-sequence batch with variable cold-block lengths exercises the
    cross-sequence pointer arithmetic of the AVX-512 kernel."""
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
    O_avx, LSE_avx = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.AVX512
    )

    atol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    rtol = 5e-3 if kv_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(O_avx, O_por, atol=atol, rtol=rtol)
    torch.testing.assert_close(LSE_avx, LSE_por, atol=1e-3, rtol=1e-3)
