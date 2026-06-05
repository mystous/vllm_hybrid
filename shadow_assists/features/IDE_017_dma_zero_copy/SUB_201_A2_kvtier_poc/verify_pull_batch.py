"""SUB_201 A2 — Batched DMA *pull* (evict) microbench.

Mirror of ``verify_batch_dma.py`` (which measures the push / fetch direction)
for the evict direction used by ``KVDramTier.evict_block``.

The previous evict-path PoC fell back to *per-layer ``pull_async``* — that is,
80 individual ``cudaMemcpyAsync`` calls per Llama-70B block, each paying the
fixed 35 μs CUDA API overhead. Llama-70B / TP=8 shape (N=80 layers × 8 KiB):

    per-layer fallback : 80 × 35 μs ≈ 2.8 ms   per block       (worst case)
    batched staged     : 1 × cudaMemcpyAsync(640 KiB) ≈ 50 μs  per block

This microbench measures both and writes a JSON dump comparable to
``verify_batch_dma.json``.

Verdict criteria:
    - ``staged`` p50 must be lower than ``fallback_per_layer`` p50.
    - speedup of at least 4× is the gate (mirror of push side regression
      resolution).
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

    # Matrix-measured baseline budget per block (D2H ≈ H2D on PCIe 5.0).
    baseline_block_us = 336.0

    pool = PinnedPool(total_limit_bytes=2 * 1024**3, numa_node=0)
    stream = pool.stream_create()

    # ── Allocate one big device source (the "GPU resident block") ────
    dev_buf = torch.full((total,), 0xA5, dtype=torch.uint8, device=dev)
    dev_base = int(dev_buf.data_ptr())
    dev_ptrs = [dev_base + i * per_layer for i in range(N)]

    # ── Pinned host destinations: 1 contiguous buffer for staged path,
    # N separate buffers for the per-layer fallback path. ───────────
    host_dst_contig = pool.alloc(total)             # staged target
    host_per_layer = [pool.alloc(per_layer) for _ in range(N)]
    host_slices_into_contig = [
        host_dst_contig + i * per_layer for i in range(N)
    ]
    sizes = [per_layer] * N

    # Pre-fill to make sure pages are committed.
    ctypes.memset(host_dst_contig, 0x00, total)
    for hp in host_per_layer:
        ctypes.memset(hp, 0x00, per_layer)

    has_staged = pool.has_pull_batch_staged()
    if not has_staged:
        raise RuntimeError(
            "libpinned_pool.so missing pinned_pool_pull_batch_async_staged — "
            "rebuild with src/pinned_pool.cpp containing the staged-pull symbol"
        )

    # ── Variant fns ──────────────────────────────────────────────────
    def fallback_per_layer():
        """Mirror of the *original* KVDramTier.evict_block fallback:
        N pull_async calls (one per layer), then a single stream_sync
        (cheaper than per-event_sync, still serializes correctness)."""
        for i in range(N):
            pool.pull_async(
                dev_ptrs[i], host_per_layer[i], per_layer, stream
            )
        pool.stream_sync(stream)

    def fallback_per_layer_event_sync():
        """Pathological worst case: per-layer pull_async + per-event sync.
        This matches what would happen if the engine called pull_async per
        layer and waited each event individually (rare but observed in the
        naïve PoC patch before stream_sync was added)."""
        for i in range(N):
            ev = pool.pull_async(
                dev_ptrs[i], host_per_layer[i], per_layer, stream
            )
            pool.event_sync(ev)
            pool.event_destroy(ev)

    def staged_pull():
        """New fast path: 1 D2H cudaMemcpyAsync of `total` bytes into the
        contiguous pinned host buffer + 1 stream_sync. host_dst_contig
        already IS the contiguous destination so we skip the fan-out
        unpack (no extra memcpy)."""
        pool.pull_batch_async_staged(
            dev_src=dev_base,
            staging_ptr=host_dst_contig,
            host_ptrs=host_slices_into_contig,
            sizes=sizes,
            stream=stream,
        )
        pool.stream_sync(stream)

    def staged_pull_with_unpack():
        """Same as staged_pull, but ALSO runs the host-side fan-out into
        N separate host destinations (the realistic case when the engine
        wants per-layer host buffers, not one contiguous slab)."""
        ev = pool.pull_batch_async_staged(
            dev_src=dev_base,
            staging_ptr=host_dst_contig,
            host_ptrs=host_per_layer,
            sizes=sizes,
            stream=stream,
        )
        pool.event_sync(ev)
        pool.event_destroy(ev)
        pool.pull_batch_unpack_staged(
            host_dst_contig, host_per_layer, sizes, event=ev,
        )

    bench_specs = [
        ("fallback_per_layer",            fallback_per_layer),
        ("fallback_per_layer_event_sync", fallback_per_layer_event_sync),
        ("staged_pull",                   staged_pull),
        ("staged_pull_with_unpack",       staged_pull_with_unpack),
    ]

    results = {}
    print()
    print(f"{'variant':<32} | {'mean μs':>8} | {'p50 μs':>8} | "
          f"{'p99 μs':>8} | {'GB/s':>6} | {'vs base':>7}")
    print("-" * 90)

    base_p50 = None
    for label, fn in bench_specs:
        _, stats = _bench(label, iters, warmup, fn)
        gbps = total / (stats["p50_us"] / 1e6) / 1024**3
        if base_p50 is None:
            base_p50 = stats["p50_us"]
            speedup = 1.0
        else:
            speedup = base_p50 / stats["p50_us"]
        results[label] = {
            **stats,
            "GBps_p50": gbps,
            "speedup_vs_fallback_per_layer": speedup,
        }
        print(
            f"{label:<32} | {stats['mean_us']:>8.2f} | {stats['p50_us']:>8.2f} | "
            f"{stats['p99_us']:>8.2f} | {gbps:>6.2f} | {speedup:>6.2f}×"
        )

    # ── Verdict ─────────────────────────────────────────────────────
    best_label, best = min(results.items(), key=lambda kv: kv[1]["p50_us"])
    print()
    print(f"best variant: {best_label}  p50 = {best['p50_us']:.2f} μs "
          f"({best['GBps_p50']:.2f} GB/s)")
    print(f"matrix baseline budget per block: {baseline_block_us:.1f} μs/evt")
    net_positive = best["p50_us"] < baseline_block_us
    speedup_vs_fallback = (
        results["fallback_per_layer"]["p50_us"] / best["p50_us"]
    )
    print(
        f"net positive vs matrix baseline? {net_positive}  "
        f"(headroom = {baseline_block_us - best['p50_us']:+.1f} μs)"
    )
    print(f"speedup vs fallback_per_layer: {speedup_vs_fallback:.2f}×")

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
            "speedup_vs_fallback_per_layer": speedup_vs_fallback,
            "gate_4x_speedup_passed": bool(speedup_vs_fallback >= 4.0),
        },
    }
    out_path = os.path.join(HERE, "verify_pull_batch.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    # cleanup
    pool.free(host_dst_contig)
    for hp in host_per_layer:
        pool.free(hp)
    del dev_buf
    torch.cuda.empty_cache()
    pool.stream_destroy(stream)
    pool.destroy()


if __name__ == "__main__":
    main()
