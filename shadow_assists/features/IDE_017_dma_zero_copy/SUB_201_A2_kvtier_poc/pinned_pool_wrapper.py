"""SUB_201 A2 PoC — standalone ctypes wrapper for libpinned_pool.so.

Goal:
  1. Provide a Python class ``PinnedPool`` usable from vllm core / engine
     paths (no torch dep required for the alloc/free hot path; torch only
     touched for DMA endpoints when the caller wants to bind a tensor).
  2. Re-verify the SUB_166 / SUB_176 numbers (DMA BW ≈ 51 GB/s @ 64 MB,
     alloc / free p50 ≤ 1 μs) using this wrapper, so KVDramTier can rely
     on it.
  3. KVCacheBlock dimensioning utility — given block_size (token count),
     num_kv_heads, head_size, layers, dtype → bytes per block per layer.

This is the *only* host-side ABI we expose to KVDramTier so the rest of
the codebase doesn't touch ctypes directly.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "build",
    "libpinned_pool.so",
)


def _load(lib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path)
    lib.pinned_pool_create.argtypes = [ctypes.c_size_t, ctypes.c_int]
    lib.pinned_pool_create.restype = ctypes.c_void_p
    lib.pinned_pool_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_destroy.restype = None
    lib.pinned_pool_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.pinned_pool_alloc.restype = ctypes.c_void_p
    lib.pinned_pool_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.pinned_pool_free.restype = None
    lib.pinned_pool_in_use.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_in_use.restype = ctypes.c_size_t
    lib.pinned_pool_owned.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_owned.restype = ctypes.c_size_t

    lib.pinned_pool_push_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.pinned_pool_push_async.restype = ctypes.c_void_p
    lib.pinned_pool_pull_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.pinned_pool_pull_async.restype = ctypes.c_void_p

    # Batched DMA — 3 variants (A: for-loop, B: native cudaMemcpyBatchAsync, C: staging).
    # Common arg shape: pool, host_ptrs[], dev_ptrs[], sizes[], n, stream
    lib.pinned_pool_push_batch_async.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.pinned_pool_push_batch_async.restype = ctypes.c_void_p
    lib.pinned_pool_push_batch_async_native.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.pinned_pool_push_batch_async_native.restype = ctypes.c_void_p
    # Staged: host_ptrs[], staging_ptr (single), dev_dst (single), sizes[], n, stream
    lib.pinned_pool_push_batch_async_staged.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.pinned_pool_push_batch_async_staged.restype = ctypes.c_void_p

    # Staged pull (mirror of staged push): D2H 1 cudaMemcpyAsync + host
    # fan-out via pull_batch_unpack_staged after event_sync.
    # Arg order: pool, dev_src, staging_ptr, host_ptrs[], sizes[], n, stream
    if hasattr(lib, "pinned_pool_pull_batch_async_staged"):
        lib.pinned_pool_pull_batch_async_staged.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.pinned_pool_pull_batch_async_staged.restype = ctypes.c_void_p
    if hasattr(lib, "pinned_pool_pull_batch_unpack_staged"):
        lib.pinned_pool_pull_batch_unpack_staged.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.pinned_pool_pull_batch_unpack_staged.restype = None

    lib.pinned_pool_event_sync.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_event_sync.restype = ctypes.c_int
    lib.pinned_pool_event_query.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_event_query.restype = ctypes.c_int
    lib.pinned_pool_event_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_event_destroy.restype = ctypes.c_int

    lib.pinned_pool_stream_create.argtypes = []
    lib.pinned_pool_stream_create.restype = ctypes.c_void_p
    lib.pinned_pool_stream_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_stream_destroy.restype = ctypes.c_int
    lib.pinned_pool_stream_sync.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_stream_sync.restype = ctypes.c_int
    return lib


@dataclass(frozen=True)
class KVBlockSpec:
    """One KV block geometry — combined K+V across all attention layers
    for a single page (block_size tokens)."""

    block_size: int            # tokens per block (e.g. 16)
    num_layers: int            # attention layers (e.g. 80 for Llama-70B)
    num_kv_heads_per_rank: int # after TP shard (Llama-70B / TP=8 → 1)
    head_size: int             # e.g. 128
    dtype_bytes: int = 2       # bf16/fp16
    kv_factor: int = 2         # K and V

    @property
    def per_layer_bytes(self) -> int:
        # block_size × num_kv_heads × head_size × kv_factor × dtype_bytes
        return (
            self.block_size
            * self.num_kv_heads_per_rank
            * self.head_size
            * self.kv_factor
            * self.dtype_bytes
        )

    @property
    def all_layers_bytes(self) -> int:
        return self.per_layer_bytes * self.num_layers


# Llama-70B reference: 80 layers, head_size=128, 8 KV heads → TP=8 ⇒ 1
# per rank. block_size=16. Per-block (all 80 layers) ≈ 80 × 16 × 1 × 128 ×
# 2 × 2 = 655 360 bytes ≈ 640 KiB.
LLAMA70B_TP8_BLOCK = KVBlockSpec(
    block_size=16,
    num_layers=80,
    num_kv_heads_per_rank=1,
    head_size=128,
)


class PinnedPool:
    """Thin OO wrapper over libpinned_pool.so.

    All methods are safe to call from the engine main loop except `destroy`
    which must run after every dependent stream has been synced.
    """

    def __init__(self, total_limit_bytes: int, numa_node: int = -1,
                 lib_path: str | None = None) -> None:
        self._lib = _load(lib_path or _DEFAULT_LIB)
        self._pool = self._lib.pinned_pool_create(
            ctypes.c_size_t(total_limit_bytes), ctypes.c_int(numa_node)
        )
        if not self._pool:
            raise RuntimeError("pinned_pool_create failed")
        # ctypes-array GC anchor for in-flight pull_batch_async_staged
        # calls (keyed by event handle). Cleared by pull_batch_unpack_staged.
        self._inflight_pull_batches: dict[int, tuple] = {}

    def alloc(self, nbytes: int) -> int:
        p = self._lib.pinned_pool_alloc(self._pool, ctypes.c_size_t(nbytes))
        if not p:
            raise MemoryError(f"pinned_pool_alloc({nbytes}) returned NULL")
        return int(p)

    def free(self, host_ptr: int) -> None:
        if not host_ptr:
            return
        self._lib.pinned_pool_free(self._pool, ctypes.c_void_p(host_ptr))

    def in_use_bytes(self) -> int:
        return int(self._lib.pinned_pool_in_use(self._pool))

    def owned_bytes(self) -> int:
        return int(self._lib.pinned_pool_owned(self._pool))

    # ── DMA primitives ─────────────────────────────────────────────────

    def stream_create(self) -> int:
        s = self._lib.pinned_pool_stream_create()
        if not s:
            raise RuntimeError("pinned_pool_stream_create failed")
        return int(s)

    def stream_destroy(self, stream: int) -> None:
        self._lib.pinned_pool_stream_destroy(ctypes.c_void_p(stream))

    def stream_sync(self, stream: int) -> None:
        self._lib.pinned_pool_stream_sync(ctypes.c_void_p(stream))

    def push_async(self, host_ptr: int, dev_ptr: int, nbytes: int,
                   stream: int) -> int:
        """Host-pinned → Device. Returns event handle."""
        ev = self._lib.pinned_pool_push_async(
            self._pool,
            ctypes.c_void_p(host_ptr),
            ctypes.c_void_p(dev_ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_void_p(stream),
        )
        return int(ev) if ev else 0

    # ── Batched DMA primitives (3 variants) ───────────────────────────
    #
    # All three return a single event handle (cudaEvent_t) covering the
    # whole batch. Caller must event_sync + event_destroy.

    @staticmethod
    def _pack_ptr_array(values):
        arr = (ctypes.c_void_p * len(values))()
        for i, v in enumerate(values):
            arr[i] = ctypes.c_void_p(int(v))
        return arr

    @staticmethod
    def _pack_size_array(values):
        arr = (ctypes.c_size_t * len(values))()
        for i, v in enumerate(values):
            arr[i] = ctypes.c_size_t(int(v))
        return arr

    def push_batch_async(self, host_ptrs, dev_ptrs, sizes, stream: int,
                         mode: str = "loop", staging_ptr: int = 0,
                         dev_dst: int = 0) -> int:
        """Submit N pinned-host → device transfers as one batch.

        mode = "loop"   → option A: for-loop cudaMemcpyAsync × N
        mode = "native" → option B: cudaMemcpyBatchAsync (CUDA 12.4+)
        mode = "staged" → option C: host-side pack into staging_ptr +
                          1 cudaMemcpyAsync(staging→dev_dst, sum(sizes))
                          (dev_ptrs argument is ignored)
        """
        n = len(sizes)
        assert n > 0, "batch must be non-empty"
        sizes_arr = self._pack_size_array(sizes)
        hosts_arr = self._pack_ptr_array(host_ptrs)
        if mode == "loop":
            devs_arr = self._pack_ptr_array(dev_ptrs)
            ev = self._lib.pinned_pool_push_batch_async(
                self._pool, hosts_arr, devs_arr, sizes_arr,
                ctypes.c_int(n), ctypes.c_void_p(stream),
            )
        elif mode == "native":
            devs_arr = self._pack_ptr_array(dev_ptrs)
            ev = self._lib.pinned_pool_push_batch_async_native(
                self._pool, hosts_arr, devs_arr, sizes_arr,
                ctypes.c_int(n), ctypes.c_void_p(stream),
            )
        elif mode == "staged":
            if not staging_ptr or not dev_dst:
                raise ValueError("staged mode requires staging_ptr and dev_dst")
            ev = self._lib.pinned_pool_push_batch_async_staged(
                self._pool, hosts_arr,
                ctypes.c_void_p(staging_ptr),
                ctypes.c_void_p(dev_dst),
                sizes_arr, ctypes.c_int(n), ctypes.c_void_p(stream),
            )
        else:
            raise ValueError(f"unknown batch mode: {mode!r}")
        return int(ev) if ev else 0

    def pull_async(self, dev_ptr: int, host_ptr: int, nbytes: int,
                   stream: int) -> int:
        """Device → Host-pinned. Returns event handle."""
        ev = self._lib.pinned_pool_pull_async(
            self._pool,
            ctypes.c_void_p(dev_ptr),
            ctypes.c_void_p(host_ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_void_p(stream),
        )
        return int(ev) if ev else 0

    # ── Batched DMA pull (option C: D2H staged) ────────────────────────
    #
    # Mirror of push_batch_async(mode="staged"). The KVDramTier evict path
    # uses this to collapse N per-layer pull_async calls into 1 single
    # cudaMemcpyAsync. The fan-out from staging buffer to per-layer host
    # destinations runs on the CPU AFTER event_sync; call
    # pull_batch_unpack_staged once the event has resolved.

    def has_pull_batch_staged(self) -> bool:
        """True if the loaded libpinned_pool.so exports the staged-pull
        symbol pair. Older builds fall back to per-layer pull_async."""
        return hasattr(self._lib, "pinned_pool_pull_batch_async_staged") and \
            hasattr(self._lib, "pinned_pool_pull_batch_unpack_staged")

    def pull_batch_async_staged(self, dev_src: int, staging_ptr: int,
                                host_ptrs, sizes, stream: int) -> int:
        """Issue 1 cudaMemcpyAsync (dev_src → staging_ptr, sum(sizes)).
        Returns an event handle; the fan-out is NOT done yet — call
        pull_batch_unpack_staged after event_sync to scatter staging to
        host_ptrs[i]."""
        if not self.has_pull_batch_staged():
            raise RuntimeError(
                "libpinned_pool.so missing pinned_pool_pull_batch_async_staged"
            )
        n = len(sizes)
        assert n > 0, "batch must be non-empty"
        sizes_arr = self._pack_size_array(sizes)
        hosts_arr = self._pack_ptr_array(host_ptrs)
        ev = self._lib.pinned_pool_pull_batch_async_staged(
            self._pool,
            ctypes.c_void_p(dev_src),
            ctypes.c_void_p(staging_ptr),
            hosts_arr,
            sizes_arr,
            ctypes.c_int(n),
            ctypes.c_void_p(stream),
        )
        # Keep the ctypes arrays alive until the caller syncs — Python
        # GC could otherwise reclaim them while the DMA / unpack still
        # needs the address table. We park them on a per-call dict
        # keyed by the event handle.
        self._inflight_pull_batches[int(ev) if ev else id(hosts_arr)] = (
            hosts_arr, sizes_arr,
        )
        return int(ev) if ev else 0

    def pull_batch_unpack_staged(self, staging_ptr: int, host_ptrs,
                                 sizes, event: int = 0) -> None:
        """CPU-side scatter from staging buffer into host_ptrs[i].
        Caller MUST have event_sync'd the event from
        pull_batch_async_staged. Optionally pass the event so the
        wrapper can release the cached ctypes arrays."""
        if not self.has_pull_batch_staged():
            raise RuntimeError(
                "libpinned_pool.so missing pinned_pool_pull_batch_unpack_staged"
            )
        n = len(sizes)
        sizes_arr = self._pack_size_array(sizes)
        hosts_arr = self._pack_ptr_array(host_ptrs)
        self._lib.pinned_pool_pull_batch_unpack_staged(
            self._pool,
            ctypes.c_void_p(staging_ptr),
            hosts_arr,
            sizes_arr,
            ctypes.c_int(n),
        )
        if event:
            self._inflight_pull_batches.pop(int(event), None)

    def event_sync(self, ev: int) -> None:
        if ev:
            self._lib.pinned_pool_event_sync(ctypes.c_void_p(ev))

    def event_query(self, ev: int) -> bool:
        if not ev:
            return True
        return self._lib.pinned_pool_event_query(ctypes.c_void_p(ev)) == 1

    def event_destroy(self, ev: int) -> None:
        if ev:
            self._lib.pinned_pool_event_destroy(ctypes.c_void_p(ev))

    def destroy(self) -> None:
        if getattr(self, "_pool", None):
            self._lib.pinned_pool_destroy(self._pool)
            self._pool = None
