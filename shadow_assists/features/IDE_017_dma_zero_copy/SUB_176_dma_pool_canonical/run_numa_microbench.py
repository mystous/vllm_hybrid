"""SUB_176 NUMA microbench — same GPU, different NUMA pool affinity.

GPU 1 (NUMA 0 attached) — pool numa=0 (same-NUMA) vs pool numa=1 (cross-NUMA).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time

LIB_PATH = os.path.join(os.path.dirname(__file__), "build", "libpinned_pool.so")
SIZES = [
    64 * 1024,
    1 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
]


def load_lib():
    lib = ctypes.CDLL(LIB_PATH)
    lib.pinned_pool_create.argtypes = [ctypes.c_size_t, ctypes.c_int]
    lib.pinned_pool_create.restype = ctypes.c_void_p
    lib.pinned_pool_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_destroy.restype = None
    lib.pinned_pool_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.pinned_pool_alloc.restype = ctypes.c_void_p
    lib.pinned_pool_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.pinned_pool_free.restype = None
    lib.pinned_pool_push_async.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p,
    ]
    lib.pinned_pool_push_async.restype = ctypes.c_void_p
    lib.pinned_pool_event_sync.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_event_sync.restype = ctypes.c_int
    lib.pinned_pool_event_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_event_destroy.restype = ctypes.c_int
    lib.pinned_pool_stream_create.argtypes = []
    lib.pinned_pool_stream_create.restype = ctypes.c_void_p
    lib.pinned_pool_stream_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_stream_destroy.restype = ctypes.c_int
    return lib


def stats(samples):
    s = sorted(samples)
    n = len(s)
    return sum(s) / n, s[n // 2], s[min(n - 1, int(n * 0.99))]


def bench_one(lib, pool, gpu_dev, sizes, iters=100):
    import torch

    stream = lib.pinned_pool_stream_create()
    results = []
    for nbytes in sizes:
        host_p = lib.pinned_pool_alloc(pool, ctypes.c_size_t(nbytes))
        ctypes.memset(host_p, 0xC3, nbytes)
        dev_buf = torch.empty(nbytes, dtype=torch.uint8, device=gpu_dev)
        for _ in range(5):
            ev = lib.pinned_pool_push_async(
                pool, ctypes.c_void_p(host_p), ctypes.c_void_p(dev_buf.data_ptr()),
                ctypes.c_size_t(nbytes), ctypes.c_void_p(stream)
            )
            lib.pinned_pool_event_sync(ctypes.c_void_p(ev))
            lib.pinned_pool_event_destroy(ctypes.c_void_p(ev))

        samples = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            ev = lib.pinned_pool_push_async(
                pool, ctypes.c_void_p(host_p), ctypes.c_void_p(dev_buf.data_ptr()),
                ctypes.c_size_t(nbytes), ctypes.c_void_p(stream)
            )
            lib.pinned_pool_event_sync(ctypes.c_void_p(ev))
            t1 = time.perf_counter_ns()
            lib.pinned_pool_event_destroy(ctypes.c_void_p(ev))
            samples.append(t1 - t0)
        m, p50, p99 = stats(samples)
        results.append({"nbytes": nbytes, "mean_us": m / 1000, "p50_us": p50 / 1000,
                        "p99_us": p99 / 1000, "bw_GBps": nbytes / (p50 / 1e9) / (1024**3)})
        lib.pinned_pool_free(pool, ctypes.c_void_p(host_p))
        del dev_buf
        torch.cuda.empty_cache()
    lib.pinned_pool_stream_destroy(ctypes.c_void_p(stream))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    lib = load_lib()

    all_results = {}
    for numa in [0, 1]:
        print(f"\n[SUB_176 NUMA] pool numa_node={numa} (GPU {args.gpu} = NUMA 0 attached)")
        pool = lib.pinned_pool_create(ctypes.c_size_t(4 * 1024**3), ctypes.c_int(numa))
        rs = bench_one(lib, pool, dev, SIZES, args.iters)
        for r in rs:
            print(f"  {r['nbytes']:>9d} B | p50 {r['p50_us']:>7.2f} μs (p99 {r['p99_us']:>7.2f}) | {r['bw_GBps']:>6.2f} GB/s")
        all_results[f"numa_{numa}"] = rs
        lib.pinned_pool_destroy(pool)

    out = os.path.join(os.path.dirname(__file__), "pool_numa_microbench.json")
    with open(out, "w") as f:
        json.dump({"gpu": args.gpu, "results": all_results}, f, indent=2)
    print(f"\n[SUB_176] NUMA wrote {out}")


if __name__ == "__main__":
    main()
