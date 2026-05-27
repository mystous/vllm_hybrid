"""SUB_176 batch microbench — batched DMA push amortization test.

Test: enqueue N transfers of same chunk size on one stream, then synchronize.
Compare effective per-transfer latency vs single-push baseline.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time

LIB_PATH = os.path.join(os.path.dirname(__file__), "build", "libpinned_pool.so")


def load_lib():
    import ctypes as ct

    lib = ct.CDLL(LIB_PATH)
    lib.pinned_pool_create.argtypes = [ct.c_size_t, ct.c_int]
    lib.pinned_pool_create.restype = ct.c_void_p
    lib.pinned_pool_destroy.argtypes = [ct.c_void_p]
    lib.pinned_pool_destroy.restype = None
    lib.pinned_pool_alloc.argtypes = [ct.c_void_p, ct.c_size_t]
    lib.pinned_pool_alloc.restype = ct.c_void_p
    lib.pinned_pool_free.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.pinned_pool_free.restype = None
    lib.pinned_pool_push_async.argtypes = [
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
    ]
    lib.pinned_pool_push_async.restype = ct.c_void_p
    lib.pinned_pool_push_batch_async.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_void_p),
        ct.POINTER(ct.c_void_p),
        ct.POINTER(ct.c_size_t),
        ct.c_int,
        ct.c_void_p,
    ]
    lib.pinned_pool_push_batch_async.restype = ct.c_void_p
    lib.pinned_pool_event_sync.argtypes = [ct.c_void_p]
    lib.pinned_pool_event_sync.restype = ct.c_int
    lib.pinned_pool_event_destroy.argtypes = [ct.c_void_p]
    lib.pinned_pool_event_destroy.restype = ct.c_int
    lib.pinned_pool_stream_create.argtypes = []
    lib.pinned_pool_stream_create.restype = ct.c_void_p
    lib.pinned_pool_stream_destroy.argtypes = [ct.c_void_p]
    lib.pinned_pool_stream_destroy.restype = ct.c_int
    lib.pinned_pool_stream_sync.argtypes = [ct.c_void_p]
    lib.pinned_pool_stream_sync.restype = ct.c_int
    return lib


def stats(samples):
    s = sorted(samples)
    n = len(s)
    return sum(s) / n, s[n // 2], s[min(n - 1, int(n * 0.99))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    lib = load_lib()
    pool = lib.pinned_pool_create(ctypes.c_size_t(4 * 1024**3), ctypes.c_int(args.numa))
    stream = lib.pinned_pool_stream_create()

    # Test: batch N transfers of 64 KB each (overhead-bound region).
    chunk = 64 * 1024
    rows = []
    for n in [1, 2, 4, 8, 16, 32]:
        # allocate n host blocks + n device blocks
        hosts = [lib.pinned_pool_alloc(pool, ctypes.c_size_t(chunk)) for _ in range(n)]
        dev_bufs = [torch.empty(chunk, dtype=torch.uint8, device=dev) for _ in range(n)]
        for h in hosts:
            ctypes.memset(h, 0xC3, chunk)

        # warmup
        for _ in range(4):
            harr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(h) for h in hosts])
            darr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(int(d.data_ptr())) for d in dev_bufs])
            sarr = (ctypes.c_size_t * n)(*[chunk] * n)
            ev = lib.pinned_pool_push_batch_async(
                pool, harr, darr, sarr, n, ctypes.c_void_p(stream)
            )
            lib.pinned_pool_event_sync(ctypes.c_void_p(ev))
            lib.pinned_pool_event_destroy(ctypes.c_void_p(ev))

        samples = []
        for _ in range(args.iters):
            harr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(h) for h in hosts])
            darr = (ctypes.c_void_p * n)(*[ctypes.c_void_p(int(d.data_ptr())) for d in dev_bufs])
            sarr = (ctypes.c_size_t * n)(*[chunk] * n)
            t0 = time.perf_counter_ns()
            ev = lib.pinned_pool_push_batch_async(
                pool, harr, darr, sarr, n, ctypes.c_void_p(stream)
            )
            lib.pinned_pool_event_sync(ctypes.c_void_p(ev))
            t1 = time.perf_counter_ns()
            lib.pinned_pool_event_destroy(ctypes.c_void_p(ev))
            samples.append(t1 - t0)

        m, p50, p99 = stats(samples)
        per_xfer = p50 / n / 1000.0  # us per transfer
        rows.append({"n": n, "chunk": chunk, "total_us_p50": p50 / 1000.0,
                     "per_xfer_us": per_xfer, "p99_us": p99 / 1000.0})
        print(f"  batch n={n:>3d} chunk={chunk} | total p50 {p50/1000:>7.2f} μs | per-xfer {per_xfer:>5.2f} μs (p99 {p99/1000:>7.2f})")

        for h in hosts:
            lib.pinned_pool_free(pool, ctypes.c_void_p(h))
        del dev_bufs
        torch.cuda.empty_cache()

    lib.pinned_pool_stream_destroy(ctypes.c_void_p(stream))
    lib.pinned_pool_destroy(pool)

    out = os.path.join(os.path.dirname(__file__), "pool_batch_microbench.json")
    with open(out, "w") as f:
        json.dump({"chunk": chunk, "rows": rows}, f, indent=2)
    print(f"[SUB_176] batch wrote {out}")


if __name__ == "__main__":
    main()
