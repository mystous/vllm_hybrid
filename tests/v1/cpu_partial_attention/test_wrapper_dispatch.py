# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for wrapper dispatch — TST_001 단계 C.

Phase 1 dev: portable C++ kernel + Python reference are wired. AMX /
AVX-512 are placeholders that cascade-fallback to portable (since
their C++ bindings are TSK_001 §4.2a/b — Phase 2 prod).

This file checks dispatch behaviour itself; numerical agreement
between portable and Python reference is checked separately in
``test_portable_cross_check.py``.

Cases:

1. ``select_isa_path()`` returns a valid :class:`ISAPath` value.
2. ``_force_path=PYTHON_REF`` is exact (atol=0, rtol=0) — same code path.
3. ``_force_path=PORTABLE`` agrees with PYTHON_REF within BF16/FP16
   tolerance (numerical floor of the C++ scalar accumulator).
4. ``_force_path=AMX`` and ``_force_path=AVX512`` cascade-fallback to
   PORTABLE on this dev machine (their kernels are not wired yet) —
   results identical to forced PORTABLE.
5. Default dispatch on this dev machine resolves to PORTABLE → same
   as forced PORTABLE.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.ops.cpu_partial_attention import (
    ISAPath,
    forward_partial_with_lse,
    python_reference_partial_attention,
    select_isa_path,
)
from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


def _make_minimal_inputs(kv_dtype, head_dim, num_kv_heads):
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=16, dtype=kv_dtype,
    )
    canonical = torch.zeros((2, layout.page_size_bytes), dtype=torch.int8)
    adapter = KVViewAdapter(canonical, layout)
    K, V = adapter.k_view(), adapter.v_view()
    gen = torch.Generator(device="cpu").manual_seed(123)
    K.copy_(torch.randn(K.shape, generator=gen, dtype=torch.float32).to(kv_dtype) * 0.5)
    V.copy_(torch.randn(V.shape, generator=gen, dtype=torch.float32).to(kv_dtype) * 0.5)

    num_q_heads = num_kv_heads * 2
    q_len = 3
    Q = torch.randn(q_len, num_q_heads, head_dim, dtype=kv_dtype) * 0.5
    n_cold = 2 * 16

    return dict(
        query=Q,
        cold_kv_cache=canonical,
        cold_kv_layout=layout,
        cold_block_ids=torch.tensor([[0, 1]], dtype=torch.int32),
        cold_block_lens=torch.tensor([2], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens_total=torch.tensor([n_cold + q_len], dtype=torch.int32),
        query_positions=torch.tensor(
            [n_cold, n_cold + 1, n_cold + 2], dtype=torch.int32,
        ),
        softmax_scale=1.0 / math.sqrt(head_dim),
    )


def test_select_isa_path_returns_valid_enum():
    path = select_isa_path()
    assert path in {
        ISAPath.AMX, ISAPath.AVX512, ISAPath.PORTABLE, ISAPath.PYTHON_REF,
    }


def test_force_python_ref_is_exact(kv_dtype, head_dim, num_kv_heads):
    """``_force_path=PYTHON_REF`` shares the same code path as the
    direct reference call → bitwise identical."""
    inputs = _make_minimal_inputs(kv_dtype, head_dim, num_kv_heads)

    O_ref, LSE_ref = python_reference_partial_attention(**inputs)
    O_w, LSE_w = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PYTHON_REF,
    )

    torch.testing.assert_close(O_w, O_ref, atol=0, rtol=0)
    torch.testing.assert_close(LSE_w, LSE_ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "forced_path", [ISAPath.PORTABLE, ISAPath.AVX512, ISAPath.AMX]
)
def test_force_simd_path_cascades_to_portable(
    kv_dtype, head_dim, num_kv_heads, forced_path
):
    """On this dev machine the AMX / AVX-512 C++ bindings are not
    wired (TSK_001 §4.2a/b — Phase 2). The wrapper must cascade
    those forced paths to the portable kernel instead. All three
    forced paths therefore produce the same result as forcing
    portable directly."""
    inputs = _make_minimal_inputs(kv_dtype, head_dim, num_kv_heads)

    O_portable, LSE_portable = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE,
    )
    O_forced, LSE_forced = forward_partial_with_lse(
        **inputs, _force_path=forced_path,
    )
    torch.testing.assert_close(O_forced, O_portable, atol=0, rtol=0)
    torch.testing.assert_close(LSE_forced, LSE_portable, atol=0, rtol=0)


def test_default_path_matches_forced_portable(
    kv_dtype, head_dim, num_kv_heads
):
    """Default dispatch on this dev machine should resolve to
    portable (no AMX hardware, AVX-512 kernel not wired) → identical
    to ``_force_path=PORTABLE``."""
    inputs = _make_minimal_inputs(kv_dtype, head_dim, num_kv_heads)

    O_default, LSE_default = forward_partial_with_lse(**inputs)
    O_portable, LSE_portable = forward_partial_with_lse(
        **inputs, _force_path=ISAPath.PORTABLE,
    )
    torch.testing.assert_close(O_default, O_portable, atol=0, rtol=0)
    torch.testing.assert_close(LSE_default, LSE_portable, atol=0, rtol=0)
