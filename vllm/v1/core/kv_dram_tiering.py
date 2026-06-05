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


_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def _is_enabled() -> bool:
    """Gate on the VLLM_KV_TIERING_DRAM env var.

    Strict allowlist: only ``1`` / ``true`` / ``yes`` / ``on`` (case
    insensitive) turn the tier on. Any other value — including unset,
    empty string, ``0``, ``no``, ``maybe``, typos — disables the
    tier, which is the safe default.
    """
    raw = os.environ.get("VLLM_KV_TIERING_DRAM", "").strip().lower()
    return raw in _ENABLED_VALUES


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
        # Phase B6 — telemetry counters (bytes-level). Gated on
        # ``VLLM_KV_TIER_TELEMETRY`` so the default path stays free of
        # extra adds. ``stats()`` always returns them so unit-tests can
        # assert them without flipping the flag.
        self._evict_bytes = 0
        self._fetch_bytes = 0

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
                "evict_bytes": self._evict_bytes,
                "fetch_bytes": self._fetch_bytes,
                # SUB_201 A2 Phase B9 — record how many distinct ptr-table
                # bindings this tier has gone through. >1 means re-bind
                # (profiling KV → real KV) actually happened; the worker
                # bind code now skips the profiling call, so the
                # boot-time value should be 1 after the B9 fix.
                "n_binds": int(getattr(self, "_n_binds", 0)),
            }

    def dump_telemetry(self, prefix: str = "[KVDramTier]") -> str:
        """Phase B6 — emit a one-line telemetry summary.

        Returns the message and writes it to stderr when
        ``VLLM_KV_TIER_TELEMETRY`` is set (truthy). Idempotent.
        """
        s = self.stats()
        msg = (
            f"{prefix} telemetry — n_evict={s['n_evict']} "
            f"n_fetch={s['n_fetch']} "
            f"evict_bytes={s['evict_bytes']} "
            f"fetch_bytes={s['fetch_bytes']} "
            f"tiered_blocks={s['tiered_blocks']} "
            f"dram_in_use={s['dram_bytes']} "
            f"skipped_full={s['n_evict_skipped_full']} "
            f"n_binds={s['n_binds']}"
        )
        raw = os.environ.get("VLLM_KV_TIER_TELEMETRY", "").strip().lower()
        if raw in _ENABLED_VALUES:
            import sys as _sys
            print(msg, file=_sys.stderr, flush=True)
        return msg

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
            self._evict_bytes += nbytes
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
            self._fetch_bytes += entry.nbytes
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

    # ──────────────────────────────────────────────────────────────
    # Engine wiring (SUB_201 A2 step 2)
    # ──────────────────────────────────────────────────────────────
    #
    # The hot paths above take a raw GPU pointer per block. In vLLM v1
    # the GPU pointer lives inside ``GPUModelRunner.kv_caches`` (a list
    # of per-layer tensors of shape ``[num_blocks, ...]``). Worker side
    # registers the per-block per-layer device pointers once via
    # ``bind_block_pointers`` so the engine hot path can look them up
    # in O(num_layers) without any torch op.
    #
    # The batched evict/fetch entries below use option C (staged) when
    # ``self._supports_staged_batch`` is true; otherwise they fall back
    # to the non-batched per-layer ``pull_async``/``push_async`` path
    # (slow but correct — the speedup is gated behind the upcoming
    # ``pinned_pool_pull_batch_async_staged`` symbol, see
    # ``DESIGN_ENGINE_WIRING.md §3``).

    def bind_block_pointers(
        self,
        per_block_layer_ptrs: list[list[int]],
        per_layer_nbytes: int,
    ) -> None:
        """Register the per-block per-layer device pointer table.

        ``per_block_layer_ptrs[b][l]`` is the GPU address of layer ``l``'s
        slice for block id ``b``. ``per_layer_nbytes`` is the byte size
        of one layer slice for one block (constant across blocks).

        SUB_201 A2 Phase B9 — **re-bindable**: a second bind (real-KV
        after the profiling-phase tiny pool) overwrites the ptr table and
        drops any previously tiered entries whose pointers now reference
        the freed profiling KV. Callers that re-bind with a *new* GPU
        pointer space MUST do so before any evict is in-flight (caller
        holds the BlockPool lock, which the profiling cleanup → real-KV
        init path satisfies by construction — both run on the
        single-threaded worker init sequence). The previous evict/fetch
        accounting counters are **kept** (they record per-process
        lifetime totals, not per-binding ones).
        """
        # Defensive bookkeeping: any block tiered against the old ptr
        # table is now stale (its host_ptr points to a buffer whose
        # twin GPU page was freed when the profiling KV cleared). Drop
        # the host buffers to recover DRAM; don't try to event_sync the
        # old pulls (they targeted GPU memory that no longer exists, but
        # the host side is fine to free immediately — the CUDA driver
        # has already torn down the source page).
        with self._lock:
            had_prior_binding = (
                getattr(self, "_per_block_layer_ptrs", None) is not None
            )
            stale_entries = list(self._table.values()) if had_prior_binding else []
            if had_prior_binding:
                self._table.clear()
                self._dram_in_use = 0
            self._per_block_layer_ptrs = per_block_layer_ptrs
            self._per_layer_nbytes = int(per_layer_nbytes)
            self._n_layers = (
                len(per_block_layer_ptrs[0]) if per_block_layer_ptrs else 0
            )
            # Phase B9 — track how many times we've re-bound so the
            # boot-time log is self-documenting.
            self._n_binds = int(getattr(self, "_n_binds", 0)) + 1
        # Free stale host buffers outside the lock (free can block).
        for entry in stale_entries:
            try:
                self._pool.free(entry.host_ptr)
            except Exception:  # pragma: no cover - defensive
                pass

    def has_pointer_binding(self) -> bool:
        return getattr(self, "_per_block_layer_ptrs", None) is not None

    def evict_block(self, block_id: int, wait: bool = True) -> bool:
        """Engine-side entry: tier the GPU block ``block_id`` whose layer
        pointers were registered via ``bind_block_pointers``.

        Fast path: when libpinned_pool.so exports
        ``pinned_pool_pull_batch_async_staged`` AND the per-layer GPU
        slabs for this block are *contiguous* (stride == per_layer_nbytes),
        we collapse the 80 per-layer ``cudaMemcpyAsync`` calls into a
        single D2H copy of size ``N × per_layer_nbytes`` → host_ptr (the
        host destination is already one contiguous pinned buffer, so the
        fan-out memcpy is a no-op — we point the staging buffer directly
        at host_ptr).

        Slow fallback (kept as graceful path): per-layer ``pull_async``
        when the symbol is absent OR per-layer pointers are non-contiguous.

        Returns True on success / already tiered, False on tier-full or
        unbound.
        """
        ptrs = getattr(self, "_per_block_layer_ptrs", None)
        if ptrs is None or block_id >= len(ptrs):
            return False
        with self._lock:
            if block_id in self._table:
                return True
            total = self._per_layer_nbytes * self._n_layers
            if self._dram_in_use + total > self._max_dram_bytes:
                self._n_evict_skipped_full += 1
                return False
            host_ptr = self._pool.alloc(total)
            block_ptrs = ptrs[block_id]
            # Detect whether the per-layer GPU slabs for THIS block are
            # contiguous; if so the staged batch becomes a single D2H copy.
            stride = self._per_layer_nbytes
            contiguous = (self._n_layers > 0) and all(
                block_ptrs[i] == block_ptrs[0] + i * stride
                for i in range(self._n_layers)
            )
            use_staged = (
                contiguous
                and hasattr(self._pool, "has_pull_batch_staged")
                and self._pool.has_pull_batch_staged()
                and hasattr(
                    self._pool._lib,
                    "pinned_pool_pull_batch_async_staged",
                )
            )
            if use_staged:
                # Single cudaMemcpyAsync DeviceToHost of `total` bytes.
                # host_ptr IS the contiguous destination, so we point the
                # staging buffer at it directly (no extra fan-out memcpy).
                host_slices = [
                    host_ptr + i * stride for i in range(self._n_layers)
                ]
                sizes = [stride] * self._n_layers
                ev = self._pool.pull_batch_async_staged(
                    dev_src=block_ptrs[0],
                    staging_ptr=host_ptr,
                    host_ptrs=host_slices,
                    sizes=sizes,
                    stream=self._stream,
                )
                # Reuse pending_ev so wait_evict / drop can serialize.
                entry = _TierEntry(
                    host_ptr=host_ptr, nbytes=total, pending_ev=ev
                )
                self._table[block_id] = entry
                self._dram_in_use += total
                self._n_evict += 1
                self._evict_bytes += total
            else:
                # Fallback: one pull per layer slice. host_ptr is contiguous
                # so per-layer offset stride == per_layer_nbytes.
                for layer_idx in range(self._n_layers):
                    self._pool.pull_async(
                        block_ptrs[layer_idx],
                        host_ptr + layer_idx * stride,
                        stride,
                        self._stream,
                    )
                entry = _TierEntry(host_ptr=host_ptr, nbytes=total, pending_ev=0)
                self._table[block_id] = entry
                self._dram_in_use += total
                self._n_evict += 1
                self._evict_bytes += total
        if wait:
            self._pool.stream_sync(self._stream)
            # For the staged fast path the unpack is a no-op (staging IS
            # the destination), but we still consume + destroy the event
            # so wait_evict doesn't see a stale handle.
            with self._lock:
                ent = self._table.get(block_id)
                if ent is not None and ent.pending_ev:
                    self._pool.event_destroy(ent.pending_ev)
                    ent.pending_ev = 0
        return True

    def fetch_block(
        self,
        block_id: int,
        wait: bool = True,
        drop_after_fetch: bool = True,
    ) -> bool:
        """Engine-side entry: push the tiered block back to HBM.

        Returns True if the push was issued (block was tiered), False
        otherwise. Same fallback semantics as ``evict_block``.
        """
        ptrs = getattr(self, "_per_block_layer_ptrs", None)
        if ptrs is None:
            return False
        with self._lock:
            entry = self._table.get(block_id)
            if entry is None:
                return False
            for layer_idx in range(self._n_layers):
                self._pool.push_async(
                    entry.host_ptr + layer_idx * self._per_layer_nbytes,
                    ptrs[block_id][layer_idx],
                    self._per_layer_nbytes,
                    self._stream,
                )
            self._n_fetch += 1
            self._fetch_bytes += entry.nbytes
        if wait:
            self._pool.stream_sync(self._stream)
        if drop_after_fetch:
            self._drop(block_id, wait=wait)
        return True

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
            # Phase B6 — register atexit hook so the process dumps the
            # final telemetry to stderr on backend kill / clean exit
            # (gated on the same VLLM_KV_TIER_TELEMETRY env flag inside
            # ``dump_telemetry``). Idempotent on re-import.
            try:
                import atexit
                atexit.register(
                    lambda: _SINGLETON is not None
                    and _SINGLETON.dump_telemetry("[KVDramTier atexit]")
                )
            except Exception:  # pragma: no cover - defensive
                pass
        return _SINGLETON


def get_existing() -> KVDramTier | None:
    return _SINGLETON


def shutdown_singleton() -> None:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            _SINGLETON.shutdown()
            _SINGLETON = None


def try_build_tier(
    max_dram_bytes: int,
    per_block_nbytes: int,
    pinned_pool_lib_path: str | None = None,
) -> KVDramTier | None:
    """Best-effort factory used by ``KVCacheManager.__init__``.

    Returns ``None`` (and logs a warning) if any of the following holds:
      * env flag VLLM_KV_TIERING_DRAM is OFF (default)
      * libpinned_pool.so cannot be loaded (missing / wrong arch)
      * PinnedPool init raises (e.g. cudaMallocHost OOM)

    This is the single entry point the engine should use to obtain a
    tier — callers must treat ``None`` as the no-op path.
    """
    if not _is_enabled():
        return None
    try:
        # Local import: the wrapper sits in shadow_assists/ and isn't on
        # sys.path during normal vllm boot. Engine wiring should arrange
        # for PYTHONPATH at startup; for the PoC smoke test we import
        # via an injected path.
        from vllm.v1.core._kv_tier_pool_loader import (  # type: ignore
            load_pinned_pool,
        )
    except Exception as e:  # pragma: no cover - exercised by smoke test
        logger.warning(
            "[KVDramTier] disabled — pool loader import failed: %s", e
        )
        return None
    try:
        pool = load_pinned_pool(
            total_limit_bytes=max_dram_bytes,
            lib_path=pinned_pool_lib_path,
        )
    except Exception as e:  # pragma: no cover - exercised by smoke test
        logger.warning(
            "[KVDramTier] disabled — PinnedPool init failed: %s", e
        )
        return None
    return get_or_create(
        pool=pool,
        max_dram_bytes=max_dram_bytes,
        per_block_nbytes=per_block_nbytes,
    )


__all__ = [
    "KVDramTier",
    "get_or_create",
    "get_existing",
    "shutdown_singleton",
    "try_build_tier",
    "_is_enabled",
]
