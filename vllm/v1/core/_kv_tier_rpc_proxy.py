# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SUB_201 A2 Phase B10 — engine-side proxy tier for cross-process RPC bind.

Background (B6 / B7 finding, MEASUREMENTS.md §7.5 / §8.1):
  * The vLLM v1 multiproc executor (TP > 1) runs ``KVCacheManager`` /
    ``BlockPool`` in the EngineCore process but the GPU pointer table
    lives in the worker process. ``bind_block_pointers`` therefore
    binds only on the worker side, so the engine-side ``KVDramTier`` has
    ``has_pointer_binding() == False`` and the BlockPool short-circuits
    every ``tier.evict_block(...)`` call → ``n_evict = 0`` and the lever
    is effectively disabled under multiproc.
  * On TP=1 (UniprocExecutor) the engine and worker share a process so
    binding + evict converge on the same KVDramTier singleton — that's
    how B7 produced ``n_evict = 512`` and B9 produced ``n_evict = 41 070``.

This module bridges the gap **without** marshaling the per-block per-layer
pointer table across IPC (which would be ~100 MB for Llama-70B TP=4):
the EngineCore holds a thin ``RpcProxyTier`` that satisfies the BlockPool's
``_kv_dram_tier`` interface (``is_tiered`` / ``has_pointer_binding`` /
``evict_block`` / ``fetch_block`` / ``drop`` / ``stats``) and forwards each
hot-path call to the worker(s) via ``MultiprocExecutor.collective_rpc``.
The worker-side handler then performs the real cudaMemcpyAsync against
its in-process ``KVDramTier`` whose pointer binding is already set up.

Activation:
  * ``VLLM_KV_TIERING_DRAM=1`` (existing flag) AND
  * ``VLLM_KV_TIER_RPC_BIND=1`` (this new flag, OFF by default).
  When the RPC flag is OFF, ``KVCacheManager`` falls back to the original
  in-process ``KVDramTier`` (TP=1 path) — no regression for callers that
  don't opt in.

RPC payload size:
  * One ``int`` (block_id) per call. The host-side cost is one ZMQ
    enqueue + one dequeue per worker × per evict/fetch.
  * For ``is_tiered``, used in BlockPool ``get_new_blocks`` /
    ``touch`` hot paths, the proxy keeps a local ``set[int]`` of
    block_ids it has issued evict to (rank 0 is the source of truth;
    cross-rank consistency is guaranteed because BlockPool replicates
    eviction decisions across all TP shards by design — each shard
    holds one shard of the same logical block).

Safety:
  * RPC failures are caught and the call returns ``False`` (== "not
    tiered" / "evict failed"); BlockPool already handles both as a
    no-op on the free / get_new_blocks paths.
  * Worker-side ``kv_tier_*`` handlers (gpu_worker.py) return defaults
    when the worker has no tier (e.g. flag mis-propagated).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_ENABLED_VALUES = {"1", "true", "yes", "on"}


def _is_rpc_bind_enabled() -> bool:
    raw = os.environ.get("VLLM_KV_TIER_RPC_BIND", "").strip().lower()
    return raw in _ENABLED_VALUES


class RpcProxyTier:
    """Engine-side stand-in for ``KVDramTier`` that forwards hot-path
    calls to worker(s) via the executor's collective_rpc.

    Implements only the subset of the KVDramTier interface BlockPool
    actually calls:
        * has_pointer_binding() -> bool
        * is_tiered(block_id) -> bool
        * evict_block(block_id, wait=True) -> bool
        * fetch_block(block_id, wait=True) -> bool
        * drop(block_id) -> None
        * stats() -> dict
        * dump_telemetry(prefix) -> str   (for atexit)
    """

    def __init__(self, model_executor: Any) -> None:
        self._executor = model_executor
        # local mirror of "which block ids are currently tiered". Updated
        # by evict_block / drop. Read by is_tiered / has_pointer_binding.
        self._tiered_ids: set[int] = set()
        self._lock = threading.Lock()
        # counters mirrored to stats() — also dumped at atexit.
        self._n_evict = 0
        self._n_fetch = 0
        self._n_drop = 0
        self._n_evict_failed = 0
        self._n_rpc_errors = 0
        # On first use: probe workers to learn whether the binding is
        # ready (== bind_block_pointers has been called inside the worker
        # tier). Cached after first True so the hot path stays cheap.
        self._binding_ready: bool | None = None

    # ──────────────────────────────────────────────────────────────
    # Internal RPC helpers
    # ──────────────────────────────────────────────────────────────

    def _rpc(self, method: str, *args, unique_reply_rank: int | None = 0):
        """Broadcast ``method`` to all workers; return result from
        ``unique_reply_rank`` (default rank 0) or list of all results.

        Captures any RPC exception → return None so callers degrade
        gracefully (the lever is purely additive — the system runs fine
        with the tier disabled).
        """
        try:
            return self._executor.collective_rpc(
                method, args=args, unique_reply_rank=unique_reply_rank,
            )
        except Exception as e:  # pragma: no cover - defensive
            self._n_rpc_errors += 1
            if self._n_rpc_errors <= 3:
                logger.warning(
                    "[KVDramTier RPC] %s failed (%s); subsequent errors "
                    "suppressed", method, e,
                )
            return None

    # ──────────────────────────────────────────────────────────────
    # BlockPool-facing interface
    # ──────────────────────────────────────────────────────────────

    def has_pointer_binding(self) -> bool:
        # Cache the True result — once workers report binding ready, it
        # stays ready for the lifetime of the engine.
        if self._binding_ready:
            return True
        result = self._rpc("kv_tier_has_pointer_binding")
        if result is True:
            self._binding_ready = True
            return True
        return False

    def is_tiered(self, block_id: int) -> bool:
        # Local mirror — engine BlockPool only asks for blocks it itself
        # tiered, so the mirror is authoritative. Avoid an RPC per call
        # (would dominate the touch / get_new_blocks hot paths).
        with self._lock:
            return block_id in self._tiered_ids

    def evict_block(self, block_id: int, wait: bool = True) -> bool:
        result = self._rpc(
            "kv_tier_evict_block", int(block_id), bool(wait),
        )
        if result is True:
            with self._lock:
                self._tiered_ids.add(int(block_id))
                self._n_evict += 1
            return True
        self._n_evict_failed += 1
        return False

    def fetch_block(
        self,
        block_id: int,
        wait: bool = True,
        drop_after_fetch: bool = True,
    ) -> bool:
        result = self._rpc(
            "kv_tier_fetch_block",
            int(block_id),
            bool(wait),
            bool(drop_after_fetch),
        )
        if result is True:
            with self._lock:
                self._n_fetch += 1
                if drop_after_fetch:
                    self._tiered_ids.discard(int(block_id))
            return True
        return False

    def drop(self, block_id: int) -> None:
        self._rpc("kv_tier_drop", int(block_id))
        with self._lock:
            self._tiered_ids.discard(int(block_id))
            self._n_drop += 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "n_evict": self._n_evict,
                "n_fetch": self._n_fetch,
                "n_drop": self._n_drop,
                "n_evict_failed": self._n_evict_failed,
                "n_rpc_errors": self._n_rpc_errors,
                "tiered_ids_engine": len(self._tiered_ids),
            }

    def dump_telemetry(self, prefix: str = "[KVDramTier proxy]") -> str:
        s = self.stats()
        msg = (
            f"{prefix} telemetry — n_evict={s['n_evict']} "
            f"n_fetch={s['n_fetch']} n_drop={s['n_drop']} "
            f"n_evict_failed={s['n_evict_failed']} "
            f"n_rpc_errors={s['n_rpc_errors']} "
            f"tiered_ids_engine={s['tiered_ids_engine']}"
        )
        raw = os.environ.get("VLLM_KV_TIER_TELEMETRY", "").strip().lower()
        if raw in _ENABLED_VALUES:
            import sys as _sys
            print(msg, file=_sys.stderr, flush=True)
        return msg


__all__ = ["RpcProxyTier", "_is_rpc_bind_enabled"]
