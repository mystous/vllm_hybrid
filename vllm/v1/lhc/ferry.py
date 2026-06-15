# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SUB_239 FERRY — "compute local, transport via DSA" for NEO KV swap-in.

The NEO CPU KV buffer (``NeoCpuKvBuffer``) may live on a NUMA node that is
*remote* to the GPU's PCIe root complex, and the swap-in read
(``copy_layer_out`` → ``.to(device)``) gathers per-request blocks via
advanced indexing — which (a) crosses UPI when the buffer is remote and
(b) loses the pinned attribute, forcing a pageable H2D fallback
(see ``gpu_model_runner.py`` SUB_030 note).

FERRY stages the gathered block tensor into a **node-local, pinned**
bounce buffer before the H2D copy. The staging copy is offloaded to Intel
DSA (``vllm.v1.lhc.dsa_lane``) when the lane is available — a CPU-free
transport, matching the microbench finding (SUB_239: DSA ferry of remote
NUMA data cut worker CPU-busy −29% / e2e −28%). When DSA is unavailable
the copy falls back to a plain ``Tensor.copy_`` — correctness is identical
(bit-exact); only the CPU-offload benefit is lost.

Env gates:
  VLLM_NEO_FERRY     = "1" : enable FERRY staging in the NEO swap-in path
                             (default off — fully reversible).
  VLLM_NEO_FERRY_MIN = int : minimum byte count to stage (default 65536).
                             Smaller gathers keep the direct path (staging
                             overhead would dominate).

Correctness note (CLAUDE.md Constraint): staging is an exact byte copy, so
the swap-in KV is identical to the non-FERRY path → distribution-equivalent
by construction (not merely similar).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_min_bytes: int | None = None


def ferry_enabled() -> bool:
    """True when ``VLLM_NEO_FERRY=1``. Cheap — read every call so a worker
    can be toggled without restart in tests."""
    return os.environ.get("VLLM_NEO_FERRY", "0") == "1"


def _ferry_min_bytes() -> int:
    global _min_bytes
    if _min_bytes is None:
        try:
            _min_bytes = int(os.environ.get("VLLM_NEO_FERRY_MIN", "65536"))
        except ValueError:
            _min_bytes = 65536
    return _min_bytes


class FerryStager:
    """Per-(shape,dtype) pinned bounce-buffer pool for FERRY staging.

    Reuses bounce buffers across calls to avoid per-step reallocation. One
    instance per worker process (the swap-in loop is single-threaded per
    worker). Buffers are first-touched by the *calling* thread, so when the
    worker is NUMA-pinned to the GPU-local node (the NEO default via
    ``VLLM_NEO_CPU_PIN_PER_WORKER``), the bounce lands GPU-local — making the
    subsequent H2D DMA source local *and* pinned.
    """

    def __init__(self) -> None:
        # key: (nbytes_bucket, dtype) → pinned 1-D byte-capacity tensor.
        self._pool: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._dsa_ops = 0
        self._cpu_ops = 0

    def _bounce(self, like: torch.Tensor) -> torch.Tensor:
        """Return a contiguous (pinned if possible) buffer shaped like
        ``like``. Reused from the pool when an equal-or-larger buffer for
        this dtype already exists."""
        nbytes = like.numel() * like.element_size()
        # Bucket to the next power-of-two to bound distinct allocations.
        bucket = 1 << (max(nbytes, 1) - 1).bit_length()
        key = (bucket, like.dtype)
        cap = self._pool.get(key)
        need_elems = bucket // like.element_size()
        if cap is None:
            try:
                cap = torch.empty(need_elems, dtype=like.dtype,
                                  pin_memory=True)
            except RuntimeError:
                # No CUDA / pinning unavailable (CPU-only host test path).
                cap = torch.empty(need_elems, dtype=like.dtype)
            self._pool[key] = cap
        # Slice + view to the exact target shape (contiguous, shares storage).
        flat = cap.narrow(0, 0, like.numel())
        return flat.view(like.shape)

    def stage(self, src: torch.Tensor) -> torch.Tensor:
        """Return a node-local, contiguous copy of ``src``.

        ``src`` is typically a ``copy_layer_out`` gather result — a
        contiguous-but-non-pinned CPU tensor that may sit on a NUMA node
        remote to the GPU. The returned tensor is contiguous and pinned
        (when CUDA is present), first-touched node-local, with identical
        contents.

        Copy is offloaded to DSA when: the lane is available, both endpoints
        are contiguous, and ``nbytes >= VLLM_NEO_FERRY_MIN``. Otherwise a
        plain ``copy_`` is used. Either way the result is bit-exact.
        """
        src_c = src.contiguous()
        dst = self._bounce(src_c)
        nbytes = src_c.numel() * src_c.element_size()

        used_dsa = False
        if nbytes >= _ferry_min_bytes():
            try:
                from vllm.v1.lhc.dsa_lane import (
                    dsa_lane_available,
                    dsa_memcpy,
                )
                if dsa_lane_available():
                    if dsa_memcpy(dst.data_ptr(), src_c.data_ptr(), nbytes):
                        used_dsa = True
            except Exception:  # noqa: BLE001
                used_dsa = False  # any failure → CPU fallback below

        if not used_dsa:
            dst.copy_(src_c)
            self._cpu_ops += 1
        else:
            self._dsa_ops += 1
        return dst

    @property
    def stats(self) -> dict[str, int]:
        return {"dsa_ops": self._dsa_ops, "cpu_ops": self._cpu_ops}
