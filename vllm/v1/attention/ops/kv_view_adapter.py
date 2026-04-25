# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV view adapter for IDE_006 (Cold-KV CPU Partial Attention).

Maps the vLLM canonical KV cache representation
(``(num_blocks, page_size_in_bytes)`` int8, see
``vllm/v1/kv_offload/spec.py:51``) to a typed K / V view (BF16 / FP16),
without copying.

This module is the §4.0 prerequisite of TSK_001 — see
``shadow_assists/features/IDE_006/TSK_001.md`` for the spec.

The adapter does **not** allocate new memory on the typical path; it
reinterprets the int8 storage with the target dtype using
:func:`torch.Tensor.view` and reshapes to the (num_blocks, block_size,
num_kv_heads, head_dim) layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVPageLayout:
    """Layout of one paged KV block (per layer) inside the canonical
    int8 representation.

    A page is composed of K and V blocks packed back-to-back. Each block
    has shape ``(block_size, num_kv_heads, head_dim)`` in the target
    dtype. The total un-padded page size in bytes is therefore::

        2 * block_size * num_kv_heads * head_dim * dtype.itemsize
    """

    head_dim: int
    num_kv_heads: int
    block_size: int
    dtype: torch.dtype

    @property
    def kv_block_bytes(self) -> int:
        """Bytes used by one of {K, V} block (half of one page)."""
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    @property
    def page_size_bytes(self) -> int:
        """Un-padded page size = 2 (K + V) blocks back-to-back."""
        return 2 * self.kv_block_bytes


class KVViewAdapter:
    """Adapter exposing typed K / V views over a canonical int8 KV tensor.

    The canonical tensor is the per-layer slice referenced by a
    :class:`vllm.v1.kv_offload.spec.CanonicalKVCacheRef` (``tensor_idx``
    + ``page_size_bytes``). This adapter assumes the caller has already
    sliced the bytes belonging to a single layer (or a homogeneous
    layer group) — i.e. ``canonical.shape == (num_blocks,
    page_size_in_bytes)`` where ``page_size_in_bytes >=
    layout.page_size_bytes``.

    The K / V views share storage with ``canonical``: writes to the
    views are reflected in ``canonical`` and vice versa. Round-trip
    (canonical → view → canonical) is therefore lossless without any
    rearrangement.

    Parameters
    ----------
    canonical:
        ``(num_blocks, page_size_in_bytes)`` ``int8`` tensor.
    layout:
        Per-layer layout of K / V inside one page.
    """

    def __init__(self, canonical: torch.Tensor, layout: KVPageLayout):
        if canonical.dtype is not torch.int8:
            raise TypeError(
                f"canonical must be int8, got {canonical.dtype}"
            )
        if canonical.dim() != 2:
            raise ValueError(
                "canonical must be a 2D tensor of shape "
                f"(num_blocks, page_size_bytes); got shape "
                f"{tuple(canonical.shape)}"
            )
        if not canonical.is_contiguous():
            raise ValueError(
                "canonical must be contiguous so K/V views can share "
                "storage without copying"
            )
        page_bytes = canonical.shape[1]
        if page_bytes < layout.page_size_bytes:
            raise ValueError(
                f"canonical page is too small: got {page_bytes} bytes, "
                f"need at least {layout.page_size_bytes} bytes for "
                f"layout {layout!r}"
            )

        self._canonical = canonical
        self.layout = layout

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        return self._canonical.shape[0]

    @property
    def page_size_bytes(self) -> int:
        return self._canonical.shape[1]

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def _slice_typed_block(self, byte_offset: int) -> torch.Tensor:
        """Extract a ``(num_blocks, block_size, num_kv_heads, head_dim)``
        view starting at ``byte_offset`` in each page.

        Implementation note: K and V are interleaved within each page
        (K block followed by V block), so a slice on dim=1 of canonical
        is **non-contiguous** across pages — a naive ``.contiguous()``
        would copy. To preserve zero-copy semantics (writes through the
        view must reach ``canonical``) we build the view explicitly via
        :func:`torch.as_strided` over the int8-→typed reinterpretation.
        """
        layout = self.layout
        dtype = layout.dtype
        if byte_offset % dtype.itemsize != 0:
            raise ValueError(
                f"byte_offset {byte_offset} is not a multiple of "
                f"dtype itemsize {dtype.itemsize}"
            )
        # Reinterpret canonical (int8) as the target dtype. Shape becomes
        # (num_blocks, page_size_bytes / itemsize), still contiguous.
        typed = self._canonical.view(dtype)
        elements_per_page = typed.shape[1]
        offset_in_elements = byte_offset // dtype.itemsize
        # Strides (in elements) of the resulting view:
        #   block dim     → elements_per_page (one full page apart)
        #   token dim     → num_kv_heads * head_dim (within a K or V block)
        #   kv_head dim   → head_dim
        #   head_dim dim  → 1
        return torch.as_strided(
            typed,
            size=(
                self.num_blocks,
                layout.block_size,
                layout.num_kv_heads,
                layout.head_dim,
            ),
            stride=(
                elements_per_page,
                layout.num_kv_heads * layout.head_dim,
                layout.head_dim,
                1,
            ),
            storage_offset=offset_in_elements,
        )

    def k_view(self) -> torch.Tensor:
        """Return the K view of all blocks.

        Shape ``(num_blocks, block_size, num_kv_heads, head_dim)`` in
        the layout dtype. K occupies the first half of each page.
        """
        return self._slice_typed_block(0)

    def v_view(self) -> torch.Tensor:
        """Return the V view of all blocks.

        Shape ``(num_blocks, block_size, num_kv_heads, head_dim)`` in
        the layout dtype. V occupies the second half of each page.
        """
        return self._slice_typed_block(self.layout.kv_block_bytes)

    def as_canonical(self) -> torch.Tensor:
        """Return the underlying int8 tensor (no copy)."""
        return self._canonical
