"""SUB_201 A2 Phase B10 — async evict (overlap) microbench.

Goal: quantify how much wall-time the evict path actually costs on the
critical path when (a) sync (current B9 behavior: wait=True →
stream_sync after each evict_block) vs (b) async (B10: wait=False; the
pending event is parked and consumed lazily). To approximate the
overlap-with-forward scenario we also run an async variant that issues
a fake "forward" kernel on the default stream BETWEEN evict_block and
wait_evict and measure the combined wall — this is the cost the engine
would pay if forward and evict truly overlapped on separate streams.

Variants:
    sync_80:
        80 × evict_block(wait=True)  — B9 hot path
    async_80_then_wait_all:
        80 × evict_block(wait=False) + stream_sync once at the end
        — measures pure issue overhead + total D2H wall
    async_80_with_forward_overlap:
        for each block: evict_block(wait=False); fake_forward();
        then wait_evict per block at the end. Fake forward is a
        modest GEMM that takes longer than the per-block evict so the
        D2H gets fully hidden — emulates the engine's "next forward
        step" hiding the spill.

Result is written to verify_async_evict.json. Verdict criterion: the
async-with-forward-overlap wall should be ≤ forward wall alone + tiny
issue overhead (i.e. the evict cost is hidden).
"""

from __future__ import annotations

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

# Reuse KVDramTier itself so we measure the same code paths the engine
# touches. Inject vllm._C stub when running outside an installed wheel.
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
try:
    from vllm.v1.core.kv_dram_tiering import KVDramTier  # noqa: E402
except ImportError:
    # If vllm._C is missing we cannot import vllm.v1; bail with a clear
    # message — the engine paths are exercised in unittest separately.
    from unittest import mock
    sys.modules.setdefault("vllm._C", mock.MagicMock())
    sys.modules.setdefault("vllm._C_stable_libtorch", mock.MagicMock())
    from vllm.v1.core.kv_dram_tiering import KVDramTier  # noqa: E402


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
    iters = int(os.environ.get("VERIFY_ITERS", "30"))
    warmup = int(os.environ.get("VERIFY_WARMUP", "3"))
    n_blocks = int(os.environ.get("VERIFY_N_BLOCKS", "80"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    spec = LLAMA70B_TP8_BLOCK
    N_layers = spec.num_layers              # 80
    per_layer = spec.per_layer_bytes        # 8192
    block_total = spec.all_layers_bytes     # 655 360 B (640 KiB)
    print(
        f"Llama-70B/TP8 shape: N_layers={N_layers}, per_layer={per_layer} B, "
        f"block_total={block_total} B ({block_total / 1024:.1f} KiB), "
        f"n_blocks={n_blocks}"
    )

    pool = PinnedPool(
        total_limit_bytes=max(4 * 1024**3, n_blocks * block_total * 2),
        numa_node=0,
    )

    # Allocate GPU source: one big buffer holding `n_blocks` blocks back-to-back.
    gpu_total = n_blocks * block_total
    dev_buf = torch.full((gpu_total,), 0xA5, dtype=torch.uint8, device=dev)
    dev_base = int(dev_buf.data_ptr())

    # Build per_block_layer_ptrs: block b layer l = dev_base + b*block_total + l*per_layer
    per_block_layer_ptrs = []
    for b in range(n_blocks):
        per_block_layer_ptrs.append(
            [dev_base + b * block_total + l * per_layer for l in range(N_layers)]
        )

    # Construct a real KVDramTier and bind the pointer table.
    tier = KVDramTier(
        pool=pool,
        max_dram_bytes=n_blocks * block_total + (16 << 20),
        per_block_nbytes=block_total,
    )
    tier.bind_block_pointers(
        per_block_layer_ptrs=per_block_layer_ptrs,
        per_layer_nbytes=per_layer,
    )

    # Fake "forward" kernel: a GEMM big enough to span the evict wall
    # so the async path's overlap can be exhibited. Tuned to roughly
    # 1.5-3 ms wall on a 3090 / H100 — large enough to dwarf 80 × 50 μs.
    K = int(os.environ.get("VERIFY_FAKE_FWD_K", "2048"))
    a = torch.randn((K, K), dtype=torch.float16, device=dev)
    b = torch.randn((K, K), dtype=torch.float16, device=dev)

    def fake_forward():
        c = torch.matmul(a, b)
        # cheap dependency to prevent dead-code elimination
        c.add_(1.0)
        torch.cuda.synchronize()

    # Warm matmul
    for _ in range(3):
        fake_forward()

    # Helper — fresh tier state per iteration (drop + re-evict). Drop
    # is cheap (releases host pinned only); the GPU is untouched.
    def reset_tier():
        # Synchronously drop any tiered entries.
        for bid in list(tier._table.keys()):
            tier._drop(bid, wait=True)

    # ── Variants ────────────────────────────────────────────────────
    def sync_80():
        reset_tier()
        for b in range(n_blocks):
            tier.evict_block(b, wait=True)

    def async_80_then_wait_all():
        reset_tier()
        for b in range(n_blocks):
            tier.evict_block(b, wait=False)
        # Single drain: all D2H ops are on tier._stream, so a single
        # stream_sync subsumes the 80 events.
        pool.stream_sync(tier._stream)

    def fwd_only():
        fake_forward()

    def sync_80_then_fwd():
        reset_tier()
        for b in range(n_blocks):
            tier.evict_block(b, wait=True)
        fake_forward()

    def async_80_with_forward_overlap():
        reset_tier()
        for b in range(n_blocks):
            tier.evict_block(b, wait=False)
        # Issue the fake forward IMMEDIATELY — runs on default stream,
        # concurrent with the D2H pulls on tier._stream.
        fake_forward()
        # Drain the tier stream after forward returns. If overlap is
        # real, drain will be near-instant because D2H completed during
        # the GEMM wall.
        pool.stream_sync(tier._stream)

    bench_specs = [
        ("fwd_only",                      fwd_only),
        ("sync_80",                       sync_80),
        ("async_80_then_wait_all",        async_80_then_wait_all),
        ("sync_80_then_fwd",              sync_80_then_fwd),
        ("async_80_with_forward_overlap", async_80_with_forward_overlap),
    ]

    results = {}
    print()
    print(
        f"{'variant':<34} | {'mean μs':>10} | {'p50 μs':>10} | "
        f"{'p99 μs':>10} | {'GB/s':>6}"
    )
    print("-" * 90)

    for label, fn in bench_specs:
        _, stats = _bench(label, iters, warmup, fn)
        # GB/s computed against the evict payload (n_blocks * block_total).
        # For fwd_only this is meaningless; skip.
        if label == "fwd_only":
            gbps = 0.0
        else:
            gbps = (n_blocks * block_total) / (stats["p50_us"] / 1e6) / 1024**3
        results[label] = {**stats, "GBps_p50": gbps}
        print(
            f"{label:<34} | {stats['mean_us']:>10.2f} | {stats['p50_us']:>10.2f} | "
            f"{stats['p99_us']:>10.2f} | {gbps:>6.2f}"
        )

    # ── Verdict ─────────────────────────────────────────────────────
    fwd_p50 = results["fwd_only"]["p50_us"]
    sync_p50 = results["sync_80"]["p50_us"]
    async_drain_p50 = results["async_80_then_wait_all"]["p50_us"]
    sync_total_p50 = results["sync_80_then_fwd"]["p50_us"]
    async_total_p50 = results["async_80_with_forward_overlap"]["p50_us"]

    # The lever's whole point: async_total ≈ max(fwd, async_drain), not
    # fwd + async_drain. Compute the % of evict cost recovered by overlap.
    sync_only_evict = sync_p50  # full evict wall, sync
    async_naive_total = fwd_p50 + sync_only_evict  # what serial costs
    overlap_savings = async_naive_total - async_total_p50
    overlap_pct = (
        100.0 * overlap_savings / async_naive_total
        if async_naive_total > 0 else 0.0
    )
    # Critical-path evict cost (sync vs async overlap)
    crit_path_sync = sync_total_p50 - fwd_p50  # what evict ADDS to fwd
    crit_path_async = async_total_p50 - fwd_p50
    hidden_pct = (
        100.0 * (crit_path_sync - crit_path_async) / crit_path_sync
        if crit_path_sync > 0 else 0.0
    )

    print()
    print(f"fwd-only p50            : {fwd_p50:>10.2f} μs")
    print(f"sync evict (alone) p50  : {sync_p50:>10.2f} μs")
    print(f"async drain (alone) p50 : {async_drain_p50:>10.2f} μs")
    print(f"sync + fwd serial p50   : {sync_total_p50:>10.2f} μs (= fwd + sync_evict)")
    print(f"async + fwd overlap p50 : {async_total_p50:>10.2f} μs")
    print(f"  → critical-path evict cost: sync={crit_path_sync:.2f} μs, async={crit_path_async:.2f} μs")
    print(f"  → evict cost hidden by overlap: {hidden_pct:.1f} %")
    print(f"  → wall savings vs sync_then_fwd: {sync_total_p50 - async_total_p50:.2f} μs")

    out = {
        "gpu": gpu,
        "iters": iters,
        "warmup": warmup,
        "n_blocks": n_blocks,
        "block_total_bytes": block_total,
        "shape": {
            "model": "Llama-70B / TP=8",
            "N_layers": N_layers,
            "per_layer_bytes": per_layer,
            "block_total_bytes": block_total,
        },
        "fake_fwd_K": K,
        "results": results,
        "verdict": {
            "fwd_only_us": fwd_p50,
            "sync_evict_us": sync_p50,
            "async_drain_us": async_drain_p50,
            "sync_plus_fwd_us": sync_total_p50,
            "async_overlap_us": async_total_p50,
            "critical_path_evict_us_sync": crit_path_sync,
            "critical_path_evict_us_async": crit_path_async,
            "evict_cost_hidden_pct": hidden_pct,
            "wall_savings_us_vs_sync": sync_total_p50 - async_total_p50,
        },
    }
    out_path = os.path.join(HERE, "verify_async_evict.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    # cleanup
    tier.shutdown()
    del dev_buf
    del a, b
    torch.cuda.empty_cache()
    pool.destroy()


if __name__ == "__main__":
    main()
