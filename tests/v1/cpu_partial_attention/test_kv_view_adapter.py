# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for ``vllm.v1.attention.ops.kv_view_adapter`` —
TST_001 단계 A (KVViewAdapter round-trip).

See ``shadow_assists/features/IDE_006/TST_001.md`` §3.1·§4.1.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.kv_view_adapter import (
    KVPageLayout,
    KVViewAdapter,
)


# ---------------------------------------------------------------------
# 단계 A — round-trip 무손실 (rtol=0, atol=0)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("num_blocks", [1, 4, 32])
def test_canonical_to_typed_view_roundtrip(
    make_canonical_kv,
    kv_dtype,
    block_size,
    head_dim,
    num_kv_heads,
    num_blocks,
):
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=kv_dtype,
    )
    canonical = make_canonical_kv(num_blocks, layout.page_size_bytes)

    adapter = KVViewAdapter(canonical, layout)

    K = adapter.k_view()
    V = adapter.v_view()

    expected_shape = (num_blocks, block_size, num_kv_heads, head_dim)
    assert K.shape == expected_shape
    assert V.shape == expected_shape
    assert K.dtype == kv_dtype
    assert V.dtype == kv_dtype

    # K / V views share storage with canonical → modifying canonical
    # must be reflected in K / V (and vice versa).
    assert K.data_ptr() == canonical.data_ptr()
    assert V.data_ptr() == canonical.data_ptr() + layout.kv_block_bytes

    # round-trip: canonical → adapter → as_canonical() returns the
    # exact same bytes.
    canonical_back = adapter.as_canonical()
    torch.testing.assert_close(canonical_back, canonical, rtol=0, atol=0)


def test_view_writes_propagate_to_canonical(
    make_canonical_kv, kv_dtype, head_dim, num_kv_heads
):
    """Writing to the typed K / V view must update the canonical bytes."""
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=16,
        dtype=kv_dtype,
    )
    num_blocks = 4
    canonical = make_canonical_kv(num_blocks, layout.page_size_bytes)
    canonical_before = canonical.clone()

    adapter = KVViewAdapter(canonical, layout)
    K = adapter.k_view()

    # Mutate via typed view
    sentinel = torch.full_like(K[0, 0, 0], fill_value=1.5)
    K[0, 0, 0].copy_(sentinel)

    # Canonical must have changed in the bytes corresponding to K[0,0,0],
    # and remain unchanged elsewhere.
    canonical_after = adapter.as_canonical()
    delta = (canonical_after.to(torch.int16) - canonical_before.to(torch.int16))
    # The K[0,0,0] slice spans `head_dim * dtype.itemsize` bytes at the
    # start of block 0's K region (block 0, token 0, head 0).
    head_bytes = head_dim * kv_dtype.itemsize
    assert delta[0, :head_bytes].abs().sum() > 0, (
        "expected canonical bytes for K[0,0,0] to be mutated"
    )
    # All other bytes must be untouched.
    delta[0, :head_bytes] = 0
    assert delta.abs().sum() == 0, (
        "writes to K[0,0,0] leaked into other regions"
    )


# ---------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------


def test_rejects_non_int8(head_dim, num_kv_heads):
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=16,
        dtype=torch.bfloat16,
    )
    bad = torch.zeros((1, layout.page_size_bytes), dtype=torch.float32)
    with pytest.raises(TypeError, match="int8"):
        KVViewAdapter(bad, layout)


def test_rejects_bad_rank(head_dim, num_kv_heads):
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=16,
        dtype=torch.bfloat16,
    )
    bad = torch.zeros((layout.page_size_bytes,), dtype=torch.int8)
    with pytest.raises(ValueError, match="2D"):
        KVViewAdapter(bad, layout)


def test_rejects_undersized_page(head_dim, num_kv_heads):
    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=16,
        dtype=torch.bfloat16,
    )
    too_small = layout.page_size_bytes - 1
    bad = torch.zeros((1, too_small), dtype=torch.int8)
    with pytest.raises(ValueError, match="too small"):
        KVViewAdapter(bad, layout)


def test_layout_size_arithmetic():
    """Sanity: page_size_bytes == 2 * block_size * num_kv_heads * head_dim
    * dtype.itemsize."""
    layout = KVPageLayout(
        head_dim=128, num_kv_heads=4, block_size=16,
        dtype=torch.bfloat16,
    )
    assert layout.kv_block_bytes == 16 * 4 * 128 * 2  # 16384
    assert layout.page_size_bytes == 2 * 16384  # 32768
