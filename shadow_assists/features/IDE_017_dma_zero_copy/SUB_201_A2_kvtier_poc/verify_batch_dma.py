"""SUB_201 A2 — Batched DMA microbench.

Goal: solve the per-layer 8 KB regression risk found in verify_dma_wrapper.json
(0.79 GB/s push BW at 8 KB → 80 layers × 9.6 μs = 768 μs per Llama-70B block,
which is *worse* than the matrix-measured 336 μs/evt memcpy baseline).

Compares three batch-DMA variants against the per-call baseline at Llama-70B
shape (N=80 layers × 8 KB per layer):

    baseline_loop      : ourselves call push_async N times, sync each event
    batch_loop  (A)    : 1 C-side for-loop of cudaMemcpyAsync + 1 event
    batch_native(B)    : 1 cudaMemcpyBatchAsync call    + 1 event (CUDA 12.4+)
    batch_staged(C)    : host-side pack into staging + 1 cudaMemcpyAsync
                         (640 KiB single contiguous transfer)

Verdict criteria (regression risk resolution):
    - batch variant must reach ≥ 13 GB/s (≥ matrix baseline 336 μs/evt =
      655 360 / 336e-6 / 2^30 ≈ 1.82 GB/s … so we use the matrix budget
      directly: total_us ≤ 336 μs/block).
    - net positive against baseline_loop required.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from pinned_pool_wrapper import (  # noqa: E402
    LLAMA70B_TP8_BLOCK,
    PinnedPool,
)


def _percentiles(samples_ns):
    s = sorted(samples_ns)
    n = len(s)
    return {
        "mean_us": statistics.fmean(s) / 1000.0,
        "p50_us": s[int(n * 0.5)] / 1000.0,
        "p90_us": s[min(n - 1, int(n * 0.9))] / 1000.0,
        "p99_us": s[min(n - 1, int(n * 0.99))] / 1000.0,
        "min_us": s[0] / 1000.0,
        "max_us": s[-1] / 1000.0,
    }


def _bench(label, iters, warmup, fn):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    return label, _percentiles(samples)


def main() -> None:
    gpu = int(os.environ.get("VERIFY_GPU", "0"))
    iters = int(os.environ.get("VERIFY_ITERS", "200"))
    warmup = int(os.environ.get("VERIFY_WARMUP", "20"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    spec = LLAMA70B_TP8_BLOCK
    N = spec.num_layers              # 80
    per_layer = spec.per_layer_bytes # 8192 B (= 8 KiB)
    total = spec.all_layers_bytes    # 655 360 B (= 640 KiB)
    print(
        f"Llama-70B/TP8 shape: N={N} layers, per_layer={per_layer} B, "
        f"total={total} B ({total / 1024:.1f} KiB)"
    )

    # Matrix-measured baseline budget per block (from main thread).
    baseline_block_us = 336.0

    pool = PinnedPool(total_limit_bytes=2 * 1024**3, numa_node=0)
    stream = pool.stream_create()

    # ── Allocate N pinned source buffers (one per layer) ────────────
    host_ptrs = [pool.alloc(per_layer) for _ in range(N)]
    for p in host_ptrs:
        ctypes.memset(p, 0x5A, per_layer)
    sizes = [per_layer] * N

    # Single big device buffer; layout it as N contiguous slabs of per_layer
    # so per-layer copies can hit dev_dst + i*per_layer.
    dev_buf = torch.empty(total, dtype=torch.uint8, device=dev)
    dev_base = int(dev_buf.data_ptr())
    dev_ptrs = [dev_base + i * per_layer for i in range(N)]

    # Staging buffer (option C) — one pinned-host contiguous block of total size.
    staging_ptr = pool.alloc(total)

    # ── Variant fns ──────────────────────────────────────────────────
    def baseline_loop():
        """Per-call push_async × N, sync each event individually.
        This mirrors what would happen if engine code naïvely called the
        single-buffer push_async per layer."""
        for i in range(N):
            ev = pool.push_async(host_ptrs[i], dev_ptrs[i], per_layer, stream)
            pool.event_sync(ev)
            pool.event_destroy(ev)

    def batch_loop():
        """Option A: 1 C-side for-loop of cudaMemcpyAsync + 1 event sync."""
        ev = pool.push_batch_async(host_ptrs, dev_ptrs, sizes, stream, mode="loop")
        pool.event_sync(ev)
        pool.event_destroy(ev)

    def batch_native():
        """Option B: cudaMemcpyBatchAsync (CUDA 12.4+) + 1 event sync."""
        ev = pool.push_batch_async(host_ptrs, dev_ptrs, sizes, stream, mode="native")
        pool.event_sync(ev)
        pool.event_destroy(ev)

    def batch_staged():
        """Option C: host-pack into staging + 1 cudaMemcpyAsync(640 KiB)."""
        ev = pool.push_batch_async(
            host_ptrs, dev_ptrs, sizes, stream,
            mode="staged", staging_ptr=staging_ptr, dev_dst=dev_base,
        )
        pool.event_sync(ev)
        pool.event_destroy(ev)

    bench_specs = [
        ("baseline_loop", baseline_loop),
        ("batch_loop_A",  batch_loop),
        ("batch_native_B", batch_native),
        ("batch_staged_C", batch_staged),
    ]

    results = {}
    print()
    print(f"{'variant':<18} | {'mean μs':>8} | {'p50 μs':>8} | "
          f"{'p99 μs':>8} | {'GB/s':>6} | {'vs base':>7}")
    print("-" * 76)

    base_p50 = None
    for label, fn in bench_specs:
        _, stats = _bench(label, iters, warmup, fn)
        gbps = total / (stats["p50_us"] / 1e6) / 1024**3
        if base_p50 is None:
            base_p50 = stats["p50_us"]
            speedup = 1.0
        else:
            speedup = base_p50 / stats["p50_us"]
        results[label] = {**stats, "GBps_p50": gbps, "speedup_vs_baseline_loop": speedup}
        print(
            f"{label:<18} | {stats['mean_us']:>8.2f} | {stats['p50_us']:>8.2f} | "
            f"{stats['p99_us']:>8.2f} | {gbps:>6.2f} | {speedup:>6.2f}×"
        )

    # ── Verdict ─────────────────────────────────────────────────────
    best_label, best = min(results.items(), key=lambda kv: kv[1]["p50_us"])
    print()
    print(f"best variant: {best_label}  p50 = {best['p50_us']:.2f} μs "
          f"({best['GBps_p50']:.2f} GB/s)")
    print(f"matrix baseline budget per block: {baseline_block_us:.1f} μs/evt")
    net_positive = best["p50_us"] < baseline_block_us
    speedup_vs_naive = results["baseline_loop"]["p50_us"] / best["p50_us"]
    print(f"net positive vs matrix baseline? {net_positive}  "
          f"(headroom = {baseline_block_us - best['p50_us']:+.1f} μs)")
    print(f"speedup vs baseline_loop: {speedup_vs_naive:.2f}×")

    out = {
        "gpu": gpu,
        "iters": iters,
        "warmup": warmup,
        "shape": {
            "model": "Llama-70B / TP=8",
            "N_layers": N,
            "per_layer_bytes": per_layer,
            "total_bytes": total,
        },
        "matrix_baseline_us_per_block": baseline_block_us,
        "results": results,
        "verdict": {
            "best_variant": best_label,
            "best_p50_us": best["p50_us"],
            "best_GBps": best["GBps_p50"],
            "net_positive_vs_matrix_baseline": bool(net_positive),
            "headroom_us": baseline_block_us - best["p50_us"],
            "speedup_vs_naive_loop": speedup_vs_naive,
        },
    }
    out_path = os.path.join(HERE, "verify_batch_dma.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    # cleanup
    for p in host_ptrs:
        pool.free(p)
    pool.free(staging_ptr)
    del dev_buf
    torch.cuda.empty_cache()
    pool.stream_destroy(stream)
    pool.destroy()


if __name__ == "__main__":
    main()
