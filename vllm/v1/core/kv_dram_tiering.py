# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SUB_201 A2 — KV DRAM tiering (cold KV → host pinned, fetch on demand).

Motivation (SUB_201 §5):
    Llama-70B suffix nsys: cudaMemcpyAsync 80.4 % of GPU API time
    (65 s in 60 s wall, 3 236 evts / s, 336 μs each) — dominated by
    inter-GPU H2D / D2H transfers under TP=8. Pure GPU residency means
    cold (old-token) KV blocks sit in HBM where they only get touched
    by the suffix attention every step, while the *active* KV could
    be served from a smaller HBM working set if cold blocks were
    parked in pinned DRAM.

Design (PoC, no engine wiring yet):

  ┌────────────────────────────────────────────────────────────────┐
  │ KVDramTier (this module)                                       │
  │                                                                 │
  │  evict_to_dram(block_id, gpu_block_ptr, nbytes, stream)         │
  │    – pool.alloc(nbytes) → host_ptr                              │
  │    – pinned_pool.pull_async(gpu_block_ptr, host_ptr, nbytes, s) │
  │    – return event; record (block_id → (host_ptr, ev))           │
  │                                                                 │
  │  fetch_to_gpu(block_id, gpu_block_ptr, nbytes, stream)          │
  │    – look up host_ptr                                           │
  │    – pinned_pool.push_async(host_ptr, gpu_block_ptr, ...)       │
  │    – schedule free-after-fetch via callback OR caller barrier   │
  │                                                                 │
  │  is_tiered(block_id) -> bool                                    │
  │  num_tiered_blocks() / dram_bytes_in_use()                      │
  └────────────────────────────────────────────────────────────────┘

Integration points (NOT done in this PoC — design only):
    1. BlockPool.free_blocks() — after ref_cnt → 0, call
       KVDramTier.evict_to_dram BEFORE FreeKVCacheBlockQueue.append_n
       (or piggyback on _maybe_evict_cached_block for cached blocks).
    2. BlockPool.get_new_blocks() / touch() — when an evicted/cached
       block hits the prefix cache, call fetch_to_gpu to repopulate
       HBM before the block is handed to the next request.
    3. Model runner must obtain a GPU pointer for the block; this means
       KVDramTier needs access to the KVCacheConfig + per-layer raw
       tensors held in GPUModelRunner.kv_caches (one tensor per layer,
       sliced per block).

Correctness barriers (CRITICAL):
    - Free path (GPU → DRAM): cudaMemcpyAsync on a non-default stream
      MUST complete BEFORE the GPU block is recycled to the next
      allocate_slots. Two options:
        (a) synchronous barrier — cudaStreamSynchronize on tier stream
            inside free_blocks (simple, low-throughput).
        (b) deferred reuse — keep the block out of free_block_queue
            until the pull event resolves (needs scheduler integration).
      PoC chooses (a) but doc (b) as next step.
    - Fetch path (DRAM → GPU): the H2D push must complete BEFORE the
      attention kernel reads that block. Two options:
        (a) sync after push_async on the engine's main stream.
        (b) issue push on the model's compute stream so it auto-
            serializes with attention.
      PoC chooses (b) — same stream, no extra barrier needed.
    - The block_id mapping is GPU-frame-of-reference (BlockPool's idx
      0..num_gpu_blocks-1). KVDramTier MUST NOT reuse the block_id
      space for DRAM accounting; it stores host_ptr per block_id.

Limitations of this PoC:
    - No engine boot / smoke (worktree venv constraint).
    - No actual per-layer slicing (caller must pass GPU pointer).
    - No async-eviction race protection beyond the barrier in
      evict_to_dram → caller must hold the BlockPool lock when calling.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    """Gate on the VLLM_KV_TIERING_DRAM env var.

    Set to non-zero / "1" / "true" to enable.
    """
    raw = os.environ.get("VLLM_KV_TIERING_DRAM", "0").strip().lower()
    return raw not in ("", "0", "false", "off")


@dataclass
class _TierEntry:
    host_ptr: int
    nbytes: int
    # last-issued event for this entry's outstanding async op (pull or
    # push). When set, holders must event_sync before reading/writing
    # the host buffer or reusing the GPU block.
    pending_ev: int = 0


class KVDramTier:
    """Per-rank singleton holding pinned-DRAM backup for evicted GPU
    KV blocks. Thread-safe by construction (single global lock — the
    hot path is alloc/free not cache lookup).
    """

    def __init__(
        self,
        pool,                          # PinnedPool from pinned_pool_wrapper
        max_dram_bytes: int,
        per_block_nbytes: int,
    ) -> None:
        self._pool = pool
        self._max_dram_bytes = int(max_dram_bytes)
        self._per_block_nbytes = int(per_block_nbytes)
        # block_id (int) → _TierEntry
        self._table: dict[int, _TierEntry] = {}
        self._lock = threading.Lock()
        # dedicated tier stream — separate from compute so eviction
        # overlaps the next forward step.
        self._stream = pool.stream_create()
        # accounting
        self._dram_in_use = 0
        # stats
        self._n_evict = 0
        self._n_fetch = 0
        self._n_evict_skipped_full = 0

    # ──────────────────────────────────────────────────────────────
    # State accessors
    # ──────────────────────────────────────────────────────────────

    def is_tiered(self, block_id: int) -> bool:
        with self._lock:
            return block_id in self._table

    def num_tiered_blocks(self) -> int:
        with self._lock:
            return len(self._table)

    def dram_bytes_in_use(self) -> int:
        with self._lock:
            return self._dram_in_use

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "n_evict": self._n_evict,
                "n_fetch": self._n_fetch,
                "n_evict_skipped_full": self._n_evict_skipped_full,
                "tiered_blocks": len(self._table),
                "dram_bytes": self._dram_in_use,
            }

    # ──────────────────────────────────────────────────────────────
    # Evict / fetch (PoC — caller passes GPU pointer; engine wiring
    # is a TODO documented in the module docstring).
    # ──────────────────────────────────────────────────────────────

    def evict_to_dram(
        self,
        block_id: int,
        gpu_block_ptr: int,
        nbytes: int | None = None,
        wait: bool = True,
    ) -> bool:
        """Spill the GPU block at `gpu_block_ptr` into pinned DRAM.

        Returns True on success, False if the tier is full.

        If `wait` is True the DMA is synced before returning — the
        caller can then immediately reuse the GPU block. For the
        async path (overlap with the next forward) set wait=False
        and call wait_evict(block_id) before reuse.
        """
        nbytes = nbytes if nbytes is not None else self._per_block_nbytes
        with self._lock:
            if block_id in self._table:
                logger.debug(
                    "[KVDramTier] evict skip — block_id=%d already tiered",
                    block_id,
                )
                return True
            if self._dram_in_use + nbytes > self._max_dram_bytes:
                self._n_evict_skipped_full += 1
                return False
            host_ptr = self._pool.alloc(nbytes)
            ev = self._pool.pull_async(
                gpu_block_ptr, host_ptr, nbytes, self._stream
            )
            entry = _TierEntry(host_ptr=host_ptr, nbytes=nbytes, pending_ev=ev)
            self._table[block_id] = entry
            self._dram_in_use += nbytes
            self._n_evict += 1
        if wait and ev:
            self._pool.event_sync(ev)
            self._pool.event_destroy(ev)
            with self._lock:
                entry.pending_ev = 0
        return True

    def wait_evict(self, block_id: int) -> None:
        """Block until the in-flight pull for `block_id` resolves so
        the GPU block can be reused safely."""
        with self._lock:
            entry = self._table.get(block_id)
            if entry is None or entry.pending_ev == 0:
                return
            ev = entry.pending_ev
        self._pool.event_sync(ev)
        self._pool.event_destroy(ev)
        with self._lock:
            entry.pending_ev = 0

    def fetch_to_gpu(
        self,
        block_id: int,
        gpu_block_ptr: int,
        wait: bool = False,
        drop_after_fetch: bool = True,
    ) -> bool:
        """Restore the tiered block back into HBM at `gpu_block_ptr`.

        Returns True if the block was tiered (push issued), False if
        the block was not in the tier (caller must regenerate via
        normal compute path).

        If `drop_after_fetch` is True, the DRAM copy is freed once the
        push event resolves. Set False when the same block is expected
        to be evicted again soon.
        """
        with self._lock:
            entry = self._table.get(block_id)
            if entry is None:
                return False
            # if the previous evict is still in flight, we must serialize
            # — push on the same stream as the pull achieves this in
            # CUDA stream semantics, so no explicit sync needed.
            ev = self._pool.push_async(
                entry.host_ptr,
                gpu_block_ptr,
                entry.nbytes,
                self._stream,
            )
            entry.pending_ev = ev
            self._n_fetch += 1
        if wait and ev:
            self._pool.event_sync(ev)
            self._pool.event_destroy(ev)
            with self._lock:
                entry.pending_ev = 0
        if drop_after_fetch:
            self._drop(block_id, wait=wait)
        return True

    def _drop(self, block_id: int, wait: bool = True) -> None:
        with self._lock:
            entry = self._table.pop(block_id, None)
            if entry is None:
                return
            self._dram_in_use -= entry.nbytes
            host_ptr = entry.host_ptr
            ev = entry.pending_ev
        if wait and ev:
            self._pool.event_sync(ev)
            self._pool.event_destroy(ev)
        self._pool.free(host_ptr)

    def drop(self, block_id: int) -> None:
        """Public: free the DRAM copy unconditionally (synchronous)."""
        self._drop(block_id, wait=True)

    def shutdown(self) -> None:
        with self._lock:
            block_ids = list(self._table.keys())
        for bid in block_ids:
            self._drop(bid, wait=True)
        if getattr(self, "_stream", None):
            self._pool.stream_destroy(self._stream)
            self._stream = 0


_SINGLETON: KVDramTier | None = None
_SINGLETON_LOCK = threading.Lock()


def get_or_create(
    pool,
    max_dram_bytes: int,
    per_block_nbytes: int,
) -> KVDramTier:
    """Per-process singleton accessor used by BlockPool when
    VLLM_KV_TIERING_DRAM is enabled."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = KVDramTier(
                pool=pool,
                max_dram_bytes=max_dram_bytes,
                per_block_nbytes=per_block_nbytes,
            )
        return _SINGLETON


def get_existing() -> KVDramTier | None:
    return _SINGLETON


def shutdown_singleton() -> None:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            _SINGLETON.shutdown()
            _SINGLETON = None


__all__ = [
    "KVDramTier",
    "get_or_create",
    "get_existing",
    "shutdown_singleton",
    "_is_enabled",
]
