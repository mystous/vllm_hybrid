# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SUB_201 A2 — KVDramTier engine-wiring smoke test.

Covers (without booting vLLM):
  1. Module import + symbol surface.
  2. ``_is_enabled`` flag parsing.
  3. ``BlockPool`` runs unchanged when ``_kv_dram_tier is None``.
  4. ``KVDramTier`` evict/fetch/drop hot path against a fake
     PinnedPool (no CUDA, no DMA — verifies the bookkeeping +
     hook firing order only).
  5. ``BlockPool.free_blocks`` fires ``evict_block`` when the tier
     is attached AND a pointer binding is registered.

Run:
    .venv/bin/python -m pytest tests/v1/spec_decode/test_kv_dram_tiering.py -v
or standalone:
    python tests/v1/spec_decode/test_kv_dram_tiering.py
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from unittest import mock


# ──────────────────────────────────────────────────────────────────────
# Bootstrap shim — the worktree under test isn't ``pip install -e``'d,
# so the compiled C extension (``vllm._C``) and the torch-touching
# platform layer are absent. Stub the minimum required for
# ``vllm.v1.core.block_pool`` to import in this worktree.
# ──────────────────────────────────────────────────────────────────────


def _stub_vllm_c() -> None:
    """Install a no-op ``vllm._C`` module so platform init succeeds."""
    if "vllm._C" in sys.modules:
        return
    import importlib

    # Pre-import vllm so attribute setattr works
    importlib.import_module("vllm")
    stub = mock.MagicMock(name="vllm._C (stub for KVDramTier smoke test)")
    sys.modules["vllm._C"] = stub
    sys.modules["vllm._C_stable_libtorch"] = mock.MagicMock(
        name="vllm._C_stable_libtorch (stub)"
    )


_stub_vllm_c()


# ──────────────────────────────────────────────────────────────────────
# Fakes — used in place of PinnedPool / CUDA so the test runs anywhere.
# ──────────────────────────────────────────────────────────────────────


class _FakePinnedPool:
    """In-process bookkeeping that mimics the PinnedPool ABI surface
    KVDramTier touches. No CUDA, no cudaMemcpy — every call is
    recorded into ``self.log`` so test assertions can inspect order."""

    def __init__(self, total_limit_bytes: int) -> None:
        self.total_limit_bytes = total_limit_bytes
        self.in_use = 0
        self.log: list[tuple] = []
        self.next_addr = 0x10_000_000
        self._next_stream = 0x70_000
        self._next_event = 0x80_000
        self._lock = threading.Lock()

    # Allocation
    def alloc(self, nbytes: int) -> int:
        with self._lock:
            ptr = self.next_addr
            self.next_addr += nbytes
            self.in_use += nbytes
            self.log.append(("alloc", nbytes, ptr))
            return ptr

    def free(self, ptr: int) -> None:
        self.log.append(("free", ptr))

    # Streams + events
    def stream_create(self) -> int:
        self._next_stream += 1
        self.log.append(("stream_create", self._next_stream))
        return self._next_stream

    def stream_destroy(self, s: int) -> None:
        self.log.append(("stream_destroy", s))

    def stream_sync(self, s: int) -> None:
        self.log.append(("stream_sync", s))

    # DMA
    def pull_async(self, dev: int, host: int, nbytes: int, stream: int) -> int:
        self._next_event += 1
        self.log.append(("pull", dev, host, nbytes, stream))
        return self._next_event

    def push_async(self, host: int, dev: int, nbytes: int, stream: int) -> int:
        self._next_event += 1
        self.log.append(("push", host, dev, nbytes, stream))
        return self._next_event

    def event_sync(self, ev: int) -> None:
        self.log.append(("event_sync", ev))

    def event_destroy(self, ev: int) -> None:
        self.log.append(("event_destroy", ev))


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestKVDramTierFlag(unittest.TestCase):
    """1) Env flag parsing — strict allowlist."""

    def test_default_off(self):
        from vllm.v1.core.kv_dram_tiering import _is_enabled

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_KV_TIERING_DRAM", None)
            self.assertFalse(_is_enabled())

    def test_truthy(self):
        from vllm.v1.core.kv_dram_tiering import _is_enabled

        for v in ("1", "true", "TRUE", "yes"):
            with mock.patch.dict(
                os.environ, {"VLLM_KV_TIERING_DRAM": v}, clear=False
            ):
                self.assertTrue(_is_enabled(), f"value {v!r} should enable")

    def test_falsy(self):
        from vllm.v1.core.kv_dram_tiering import _is_enabled

        for v in ("0", "false", "off", "", "no"):
            with mock.patch.dict(
                os.environ, {"VLLM_KV_TIERING_DRAM": v}, clear=False
            ):
                self.assertFalse(_is_enabled(), f"value {v!r} should disable")


class TestKVDramTierHotPath(unittest.TestCase):
    """2) Evict / fetch / drop bookkeeping against fake pool."""

    def setUp(self):
        from vllm.v1.core.kv_dram_tiering import KVDramTier, shutdown_singleton

        shutdown_singleton()
        self.pool = _FakePinnedPool(total_limit_bytes=16 << 20)
        self.tier = KVDramTier(
            pool=self.pool,
            max_dram_bytes=16 << 20,
            per_block_nbytes=4096,
        )
        # bind a 4-block × 2-layer table so evict_block/fetch_block work
        self.per_layer_bytes = 1024
        per_block_ptrs = [
            [0xA000 + b * 0x10000 + l * 0x1000 for l in range(2)]
            for b in range(4)
        ]
        self.tier.bind_block_pointers(
            per_block_layer_ptrs=per_block_ptrs,
            per_layer_nbytes=self.per_layer_bytes,
        )

    def tearDown(self):
        from vllm.v1.core.kv_dram_tiering import shutdown_singleton

        self.tier.shutdown()
        shutdown_singleton()

    def test_initial_state(self):
        self.assertEqual(self.tier.num_tiered_blocks(), 0)
        self.assertEqual(self.tier.dram_bytes_in_use(), 0)
        self.assertTrue(self.tier.has_pointer_binding())

    def test_evict_then_fetch_then_drop(self):
        ok = self.tier.evict_block(0, wait=True)
        self.assertTrue(ok)
        self.assertEqual(self.tier.num_tiered_blocks(), 1)
        # two layer slices = two pulls; one alloc; one stream_sync (wait=True)
        kinds = [e[0] for e in self.pool.log]
        self.assertIn("alloc", kinds)
        self.assertEqual(kinds.count("pull"), 2)
        self.assertIn("stream_sync", kinds)

        # double evict is a no-op (already tiered → True, no extra alloc)
        before = len(self.pool.log)
        ok = self.tier.evict_block(0, wait=True)
        self.assertTrue(ok)
        self.assertEqual(self.tier.num_tiered_blocks(), 1)
        self.assertEqual(len(self.pool.log), before)

        # fetch + drop
        ok = self.tier.fetch_block(0, wait=True, drop_after_fetch=True)
        self.assertTrue(ok)
        self.assertEqual(self.tier.num_tiered_blocks(), 0)
        kinds = [e[0] for e in self.pool.log]
        self.assertEqual(kinds.count("push"), 2)
        self.assertIn("free", kinds)

    def test_evict_capacity_full(self):
        # max_dram_bytes is 16 MiB; per block = 2 layers × 1024 = 2 KiB.
        # Shrink the tier in-place for a tighter test.
        self.tier._max_dram_bytes = 2048  # exactly one block
        self.assertTrue(self.tier.evict_block(0, wait=True))
        self.assertFalse(self.tier.evict_block(1, wait=True))
        stats = self.tier.stats()
        self.assertEqual(stats["n_evict"], 1)
        self.assertEqual(stats["n_evict_skipped_full"], 1)

    def test_fetch_unknown_block(self):
        self.assertFalse(
            self.tier.fetch_block(99, wait=True, drop_after_fetch=True)
        )

    def test_unbound_returns_false(self):
        from vllm.v1.core.kv_dram_tiering import KVDramTier

        unbound = KVDramTier(
            pool=_FakePinnedPool(1 << 20),
            max_dram_bytes=1 << 20,
            per_block_nbytes=4096,
        )
        self.assertFalse(unbound.has_pointer_binding())
        self.assertFalse(unbound.evict_block(0))
        self.assertFalse(unbound.fetch_block(0))
        unbound.shutdown()

    def test_rebind_expands_ptr_table_b9(self):
        """SUB_201 A2 Phase B9 — re-bind with a *larger* ptr table (e.g.
        profiling-KV 4 blocks → real-KV 32 blocks) must

          1. expand the addressable block_id range, and
          2. drop any host buffers tiered against the previous binding
             (their twin GPU page no longer exists), resetting
             ``dram_bytes_in_use`` and ``num_tiered_blocks``,
          3. NOT reset the lifetime evict/fetch counters
             (``n_evict`` / ``n_fetch``), and
          4. increment ``n_binds``.
        """
        # Step 1 — evict 1 block against the initial 4-block binding.
        self.assertTrue(self.tier.evict_block(2, wait=True))
        s1 = self.tier.stats()
        self.assertEqual(s1["n_evict"], 1)
        self.assertEqual(s1["tiered_blocks"], 1)
        self.assertEqual(s1["n_binds"], 1)
        bytes_before = s1["dram_bytes"]
        self.assertGreater(bytes_before, 0)

        # Step 2 — re-bind with a 32-block table (simulates real-KV
        # init after the profiling KV is torn down).
        per_block_ptrs_2 = [
            [0xB000 + b * 0x10000 + l * 0x1000 for l in range(2)]
            for b in range(32)
        ]
        self.tier.bind_block_pointers(
            per_block_layer_ptrs=per_block_ptrs_2,
            per_layer_nbytes=self.per_layer_bytes,
        )
        s2 = self.tier.stats()
        # bind counter ticked + stale entries dropped + accounting reset
        self.assertEqual(s2["n_binds"], 2)
        self.assertEqual(s2["tiered_blocks"], 0)
        self.assertEqual(s2["dram_bytes"], 0)
        # lifetime n_evict preserved (it's a counter, not a gauge)
        self.assertEqual(s2["n_evict"], 1)

        # Step 3 — block_id beyond the *old* 4-block range now evicts.
        self.assertTrue(self.tier.evict_block(17, wait=True))
        s3 = self.tier.stats()
        self.assertEqual(s3["n_evict"], 2)
        self.assertEqual(s3["tiered_blocks"], 1)

    def test_rebind_idempotent_no_double_alloc_b9(self):
        """SUB_201 A2 Phase B9 — re-binding the *same* ptr table (no
        topology change) must not double-count evict bytes or fail."""
        self.assertTrue(self.tier.evict_block(0, wait=True))
        self.assertTrue(self.tier.evict_block(1, wait=True))
        s1 = self.tier.stats()
        self.assertEqual(s1["tiered_blocks"], 2)

        # Re-bind with the same shape — entries are still dropped
        # (defensive; the worker code never re-binds without a topology
        # change, but the bookkeeping must remain self-consistent).
        per_block_ptrs_again = [
            [0xA000 + b * 0x10000 + l * 0x1000 for l in range(2)]
            for b in range(4)
        ]
        self.tier.bind_block_pointers(
            per_block_layer_ptrs=per_block_ptrs_again,
            per_layer_nbytes=self.per_layer_bytes,
        )
        s2 = self.tier.stats()
        self.assertEqual(s2["n_binds"], 2)
        self.assertEqual(s2["tiered_blocks"], 0)
        self.assertEqual(s2["dram_bytes"], 0)
        # n_evict counter is the lifetime total, NOT reset by re-bind.
        self.assertEqual(s2["n_evict"], 2)


class TestBlockPoolNoOpWhenTierNone(unittest.TestCase):
    """3) Default BlockPool path: tier ref is None → all hooks no-op."""

    def test_free_blocks_no_tier(self):
        from vllm.v1.core.block_pool import BlockPool

        pool = BlockPool(
            num_gpu_blocks=8,
            enable_caching=False,
            hash_block_size=16,
        )
        self.assertIsNone(pool._kv_dram_tier)
        # Just exercise the path: alloc 2 blocks then free them
        blks = pool.get_new_blocks(2)
        pool.free_blocks(blks)
        self.assertEqual(pool.get_num_free_blocks(), 7)


class TestBlockPoolHooksWithFakeTier(unittest.TestCase):
    """4) When a tier with pointer binding is attached, free_blocks
    invokes evict_block on cached blocks; get_new_blocks invokes drop
    on tiered re-allocated blocks; touch invokes fetch_block on tiered
    cache hits."""

    def setUp(self):
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_dram_tiering import KVDramTier, shutdown_singleton

        shutdown_singleton()
        self.fake_pool = _FakePinnedPool(total_limit_bytes=1 << 20)
        self.tier = KVDramTier(
            pool=self.fake_pool,
            max_dram_bytes=1 << 20,
            per_block_nbytes=1024,
        )
        # bind 8 fake blocks × 1 layer
        self.tier.bind_block_pointers(
            per_block_layer_ptrs=[[0xBA000 + b * 0x1000] for b in range(8)],
            per_layer_nbytes=1024,
        )
        self.pool = BlockPool(
            num_gpu_blocks=8,
            enable_caching=True,
            hash_block_size=16,
            kv_dram_tier=self.tier,
        )

    def tearDown(self):
        from vllm.v1.core.kv_dram_tiering import shutdown_singleton

        self.tier.shutdown()
        shutdown_singleton()

    @staticmethod
    def _stamp_cached(block, group_id: int = 0) -> None:
        """Pretend the block is cached so the evict policy fires.

        ``BlockHashWithGroupId`` is a ``NewType(bytes)`` — the
        free_blocks evict policy only checks ``block_hash is not None``,
        so any non-empty bytes string works.
        """
        block.block_hash = b"dummy_hash_xxxxxx" + group_id.to_bytes(
            4, "big", signed=False
        )

    def test_free_blocks_evicts_cached_to_dram(self):
        blks = self.pool.get_new_blocks(2)
        for b in blks:
            self._stamp_cached(b)
        before = self.tier.num_tiered_blocks()
        self.pool.free_blocks(blks)
        after = self.tier.num_tiered_blocks()
        self.assertEqual(after - before, 2, msg=f"log={self.fake_pool.log[-12:]}")
        kinds = [e[0] for e in self.fake_pool.log]
        # one alloc + one pull (single layer) + one stream_sync per evict
        self.assertEqual(kinds.count("alloc"), 2)
        self.assertEqual(kinds.count("pull"), 2)

    def test_free_blocks_skips_uncached(self):
        blks = self.pool.get_new_blocks(2)
        # do NOT stamp — block_hash stays None
        self.pool.free_blocks(blks)
        self.assertEqual(self.tier.num_tiered_blocks(), 0)

    def test_get_new_blocks_drops_tiered_dram_copy(self):
        # Drain the LRU queue so the freed (and tiered) block is the
        # next one returned by popleft_n. BlockPool ctor consumes one
        # null block, so 7 user blocks remain initially.
        reserved = self.pool.get_new_blocks(7)
        # The block we'll tier:
        target = reserved[0]
        self._stamp_cached(target)
        # Free *only* the target; the other 6 stay ref_cnt=1.
        self.pool.free_blocks([target])
        self.assertEqual(self.tier.num_tiered_blocks(), 1)
        # Now the LRU queue holds exactly one block — the tiered one.
        # get_new_blocks(1) must return it and drop the DRAM copy.
        reused = self.pool.get_new_blocks(1)
        self.assertIs(reused[0], target)
        self.assertEqual(self.tier.num_tiered_blocks(), 0)
        self.assertEqual(reused[0].ref_cnt, 1)

    def test_touch_fetches_tiered_cache_hit(self):
        blks = self.pool.get_new_blocks(1)
        for b in blks:
            self._stamp_cached(b)
        self.pool.free_blocks(blks)
        self.assertEqual(self.tier.num_tiered_blocks(), 1)
        # touch — simulates a prefix-cache hit re-attach
        before_push = sum(1 for e in self.fake_pool.log if e[0] == "push")
        self.pool.touch(blks)
        after_push = sum(1 for e in self.fake_pool.log if e[0] == "push")
        self.assertGreater(after_push, before_push)
        # touch also drops by default; tier should be empty
        self.assertEqual(self.tier.num_tiered_blocks(), 0)


# ──────────────────────────────────────────────────────────────────────
# Standalone runner (for environments without pytest collection)
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Run all test classes in this module sequentially with verbose output.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
