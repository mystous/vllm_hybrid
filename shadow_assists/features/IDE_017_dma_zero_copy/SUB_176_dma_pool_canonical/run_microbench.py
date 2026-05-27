"""SUB_176 — Pinned pool microbench (reproduce SUB_166 with pool API).

SUB_166 protocol: torch pin_memory + .copy_(non_blocking=True) + cuda stream sync.
SUB_176: same protocol but allocator = our libpinned_pool.so (lockless ring pool).

Goal:
  1. Reproduce SUB_166 latency curve (35 μs / 1 MB crossover / 53.6 GB/s).
  2. Measure pool alloc/free overhead (< 5 μs target).
  3. Per-size sweep 4 KB .. 64 MB.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time

LIB_PATH = os.path.join(os.path.dirname(__file__), "build", "libpinned_pool.so")
DEFAULT_SIZES_BYTES = [
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1 * 1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
]


def load_lib():
    lib = ctypes.CDLL(LIB_PATH)
    # Signatures
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
    lib.pinned_pool_class_capacity.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.pinned_pool_class_capacity.restype = ctypes.c_size_t
    lib.pinned_pool_class_block_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.pinned_pool_class_block_size.restype = ctypes.c_size_t
    lib.pinned_pool_pick_size_class.argtypes = [ctypes.c_size_t]
    lib.pinned_pool_pick_size_class.restype = ctypes.c_int

    lib.pinned_pool_push_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
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
    lib.pinned_pool_stream_sync.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_stream_sync.restype = ctypes.c_int
    return lib


def stats(samples):
    n = len(samples)
    s = sorted(samples)
    mean = sum(s) / n
    p50 = s[int(n * 0.5)]
    p99 = s[min(n - 1, int(n * 0.99))]
    return mean, p50, p99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=1, help="GPU index (NUMA-attached one preferred)")
    parser.add_argument("--total-limit", type=int, default=4 * 1024 * 1024 * 1024, help="pool size (bytes)")
    parser.add_argument("--numa", type=int, default=0, help="NUMA node hint")
    parser.add_argument("--out-json", type=str, default="pool_microbench.json")
    args = parser.parse_args()

    # cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    print(f"[SUB_176] device = {torch.cuda.get_device_name(0)} (GPU index in driver = {args.gpu})")

    lib = load_lib()
    pool = lib.pinned_pool_create(ctypes.c_size_t(args.total_limit), ctypes.c_int(args.numa))
    assert pool, "pool create failed"
    print(f"[SUB_176] pool created — owned {lib.pinned_pool_owned(pool) / (1024**2):.1f} MB")
    for c in range(5):
        bs = lib.pinned_pool_class_block_size(pool, c)
        cap = lib.pinned_pool_class_capacity(pool, c)
        print(f"  class {c}: block {bs/1024:.0f} KB × {cap}")

    stream = lib.pinned_pool_stream_create()
    assert stream

    rows = []
    for nbytes in DEFAULT_SIZES_BYTES:
        # ─── (A) alloc/free overhead (pool hot-path) ───
        alloc_ns = []
        free_ns = []
        warmup = 8
        cnt = max(args.iters, 100)
        for i in range(cnt + warmup):
            t0 = time.perf_counter_ns()
            p = lib.pinned_pool_alloc(pool, ctypes.c_size_t(nbytes))
            t1 = time.perf_counter_ns()
            lib.pinned_pool_free(pool, ctypes.c_void_p(p))
            t2 = time.perf_counter_ns()
            if i >= warmup:
                alloc_ns.append(t1 - t0)
                free_ns.append(t2 - t1)

        a_mean, a_p50, a_p99 = stats(alloc_ns)
        f_mean, f_p50, f_p99 = stats(free_ns)

        # ─── (B) DMA push latency (host→device via pool.push_async) ───
        # acquire one pool block once, fill with deterministic data
        host_p = lib.pinned_pool_alloc(pool, ctypes.c_size_t(nbytes))
        assert host_p, f"pool exhausted for {nbytes}"
        # GPU-side dst
        dev_buf = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        # init pinned with a pattern (via ctypes memset)
        ctypes.memset(host_p, 0xA5, nbytes)

        # warmup
        for _ in range(8):
            ev = lib.pinned_pool_push_async(
                pool, ctypes.c_void_p(host_p), ctypes.c_void_p(dev_buf.data_ptr()),
                ctypes.c_size_t(nbytes), ctypes.c_void_p(stream),
            )
            lib.pinned_pool_event_sync(ctypes.c_void_p(ev))
            lib.pinned_pool_event_destroy(ctypes.c_void_p(ev))

        dma_ns = []
        for _ in range(args.iters):
            t0 = time.perf_counter_ns()
            ev = lib.pinned_pool_push_async(
                pool, ctypes.c_void_p(host_p), ctypes.c_void_p(dev_buf.data_ptr()),
                ctypes.c_size_t(nbytes), ctypes.c_void_p(stream),
            )
            lib.pinned_pool_event_sync(ctypes.c_void_p(ev))
            t1 = time.perf_counter_ns()
            lib.pinned_pool_event_destroy(ctypes.c_void_p(ev))
            dma_ns.append(t1 - t0)

        d_mean, d_p50, d_p99 = stats(dma_ns)

        # correctness check — copy back and compare
        host_back = lib.pinned_pool_alloc(pool, ctypes.c_size_t(nbytes))
        lib.pinned_pool_stream_sync(ctypes.c_void_p(stream))
        # Use torch tensor view of dev_buf to copy back via cudaMemcpy (we have no pull api here, do simple device→host)
        out = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
        out.copy_(dev_buf.cpu())  # implicit copy
        ok = bool((out == 0xA5).all().item())

        lib.pinned_pool_free(pool, ctypes.c_void_p(host_back))
        lib.pinned_pool_free(pool, ctypes.c_void_p(host_p))
        del dev_buf
        torch.cuda.empty_cache()

        bw = nbytes / (d_p50 / 1e9) / (1024**3)
        regime = "overhead" if nbytes <= 256 * 1024 else ("crossover" if nbytes == 1024 * 1024 else "bandwidth")
        rows.append({
            "nbytes": nbytes,
            "alloc_us": a_p50 / 1000.0,
            "alloc_p99_us": a_p99 / 1000.0,
            "free_us": f_p50 / 1000.0,
            "free_p99_us": f_p99 / 1000.0,
            "dma_mean_us": d_mean / 1000.0,
            "dma_p50_us": d_p50 / 1000.0,
            "dma_p99_us": d_p99 / 1000.0,
            "bandwidth_GBps": bw,
            "regime": regime,
            "roundtrip_ok": ok,
        })
        print(
            f"  {nbytes:>9d} B | alloc p50 {a_p50/1000:>6.2f} μs (p99 {a_p99/1000:>6.2f}) "
            f"| free p50 {f_p50/1000:>6.2f} μs | DMA p50 {d_p50/1000:>7.2f} μs (p99 {d_p99/1000:>7.2f}) "
            f"| {bw:>6.2f} GB/s | rt_ok={ok}"
        )

    lib.pinned_pool_stream_destroy(ctypes.c_void_p(stream))
    lib.pinned_pool_destroy(pool)

    out_path = os.path.join(os.path.dirname(__file__), args.out_json)
    with open(out_path, "w") as f:
        json.dump({"iters": args.iters, "gpu": args.gpu, "rows": rows}, f, indent=2)
    print(f"[SUB_176] wrote {out_path}")


if __name__ == "__main__":
    main()
