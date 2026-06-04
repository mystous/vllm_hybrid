"""Quick standalone test that re-confirms libpinned_pool.so DMA
performance using the Python wrapper KVDramTier will rely on."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from pinned_pool_wrapper import (  # noqa: E402
    LLAMA70B_TP8_BLOCK,
    PinnedPool,
)


def _stats(samples):
    n = len(samples)
    s = sorted(samples)
    return {
        "mean_us": sum(s) / n / 1000.0,
        "p50_us": s[int(n * 0.5)] / 1000.0,
        "p99_us": s[min(n - 1, int(n * 0.99))] / 1000.0,
    }


def main() -> None:
    gpu = int(os.environ.get("VERIFY_GPU", "1"))
    iters = int(os.environ.get("VERIFY_ITERS", "100"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    pool = PinnedPool(total_limit_bytes=4 * 1024**3, numa_node=0)
    stream = pool.stream_create()

    sizes = {
        "4KB": 4 * 1024,
        "64KB": 64 * 1024,
        "1MB": 1024**2,
        "16MB": 16 * 1024**2,
        "64MB": 64 * 1024**2,
        "llama70b_tp8_block": LLAMA70B_TP8_BLOCK.all_layers_bytes,
        "llama70b_tp8_per_layer": LLAMA70B_TP8_BLOCK.per_layer_bytes,
    }

    rows = {}
    for name, nbytes in sizes.items():
        host_ptr = pool.alloc(nbytes)
        ctypes.memset(host_ptr, 0x5A, nbytes)
        dev_buf = torch.empty(nbytes, dtype=torch.uint8, device=dev)

        # warmup
        for _ in range(8):
            ev = pool.push_async(host_ptr, dev_buf.data_ptr(), nbytes, stream)
            pool.event_sync(ev)
            pool.event_destroy(ev)

        push_ns, pull_ns = [], []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            ev = pool.push_async(host_ptr, dev_buf.data_ptr(), nbytes, stream)
            pool.event_sync(ev)
            t1 = time.perf_counter_ns()
            pool.event_destroy(ev)
            push_ns.append(t1 - t0)
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            ev = pool.pull_async(dev_buf.data_ptr(), host_ptr, nbytes, stream)
            pool.event_sync(ev)
            t1 = time.perf_counter_ns()
            pool.event_destroy(ev)
            pull_ns.append(t1 - t0)

        ps, pl = _stats(push_ns), _stats(pull_ns)
        push_bw = nbytes / (ps["p50_us"] / 1e6) / 1024**3
        pull_bw = nbytes / (pl["p50_us"] / 1e6) / 1024**3
        rows[name] = {
            "nbytes": nbytes,
            "push": ps,
            "push_bw_GBps": push_bw,
            "pull": pl,
            "pull_bw_GBps": pull_bw,
        }
        print(
            f"  {name:>22s} | {nbytes:>9d} B | "
            f"push p50 {ps['p50_us']:>7.2f} μs / {push_bw:>5.2f} GB/s | "
            f"pull p50 {pl['p50_us']:>7.2f} μs / {pull_bw:>5.2f} GB/s"
        )
        pool.free(host_ptr)
        del dev_buf
        torch.cuda.empty_cache()

    out = {
        "gpu": gpu,
        "iters": iters,
        "pool_owned_MB": pool.owned_bytes() / 1024**2,
        "rows": rows,
    }
    out_path = os.path.join(HERE, "verify_dma_wrapper.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    pool.stream_destroy(stream)
    pool.destroy()


if __name__ == "__main__":
    main()
