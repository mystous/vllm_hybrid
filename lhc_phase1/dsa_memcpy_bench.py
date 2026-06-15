"""
LHC Phase 1 — DSA memcpy microbench (CPU/GPU baselines).

This script measures memcpy bandwidth + latency for KV-cache-eviction-shaped
transfers. The DSA backend itself requires kernel WQ enable (accel-config
enable-device dsa0 + enable-wq), which is a system-wide change and may be
gated by the harness; if /dev/dsa/wq0.0 is absent we mark DSA as N/A.

Backends:
  (a) DSA descriptor ring  — requires /dev/dsa/wq*; see dsa_memcpy_bench.c
  (b) glibc memcpy         — pure CPU
  (c) cudaMemcpyAsync H2D  — pinned host -> device
  (d) cudaMemcpyAsync D2H  — device -> pinned host

Sizes: 4KB, 64KB, 1MB, 16MB, 256MB
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import time
from pathlib import Path

import torch

SIZES = [
    ("4KB", 4 * 1024),
    ("64KB", 64 * 1024),
    ("1MB", 1 * 1024 * 1024),
    ("16MB", 16 * 1024 * 1024),
    ("256MB", 256 * 1024 * 1024),
]

WARMUP = 5
# Per-size iterations: smaller for large buffers to keep wall-time tractable.
def iters_for(size: int) -> int:
    if size <= 64 * 1024:
        return 2000
    if size <= 1 * 1024 * 1024:
        return 500
    if size <= 16 * 1024 * 1024:
        return 200
    return 30


libc = ctypes.CDLL("libc.so.6")
libc.memcpy.restype = ctypes.c_void_p
libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def _alloc_aligned(size: int) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
    """Returns (ptr, free_handle) where free_handle is a Python object whose
    lifetime owns the buffer."""
    # use bytearray + ctypes view (aligned by glibc malloc, 16B+)
    buf = (ctypes.c_ubyte * size)()
    ptr = ctypes.cast(buf, ctypes.c_void_p)
    return ptr, buf


def bench_glibc_memcpy(size: int) -> dict:
    src_ptr, src_h = _alloc_aligned(size)
    dst_ptr, dst_h = _alloc_aligned(size)
    # Fill src
    ctypes.memset(src_ptr, 0xA5, size)
    iters = iters_for(size)
    for _ in range(WARMUP):
        libc.memcpy(dst_ptr, src_ptr, size)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        libc.memcpy(dst_ptr, src_ptr, size)
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    samples.sort()
    med_ns = samples[len(samples) // 2]
    bw_gbs = size / med_ns  # bytes / ns = GB/s (10^9)
    return {
        "backend": "glibc_memcpy",
        "size_bytes": size,
        "lat_us_p50": round(med_ns / 1000, 3),
        "bw_GBs_p50": round(bw_gbs, 2),
        "iters": iters,
    }


def bench_cuda_h2d(size: int) -> dict:
    # Pinned host buffer
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host.fill_(0xA5)
    dev = torch.empty(size, dtype=torch.uint8, device="cuda:0")
    stream = torch.cuda.Stream(device="cuda:0")
    iters = iters_for(size)
    for _ in range(WARMUP):
        with torch.cuda.stream(stream):
            dev.copy_(host, non_blocking=True)
        stream.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            dev.copy_(host, non_blocking=True)
            end.record(stream)
        end.synchronize()
        ms = start.elapsed_time(end)
        samples.append(ms * 1e6)  # ns
    samples.sort()
    med_ns = samples[len(samples) // 2]
    bw_gbs = size / med_ns
    return {
        "backend": "cuda_memcpy_async_h2d",
        "size_bytes": size,
        "lat_us_p50": round(med_ns / 1000, 3),
        "bw_GBs_p50": round(bw_gbs, 2),
        "iters": iters,
    }


def bench_cuda_d2h(size: int) -> dict:
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(size, dtype=torch.uint8, device="cuda:0")
    dev.fill_(0xA5)
    stream = torch.cuda.Stream(device="cuda:0")
    iters = iters_for(size)
    for _ in range(WARMUP):
        with torch.cuda.stream(stream):
            host.copy_(dev, non_blocking=True)
        stream.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            host.copy_(dev, non_blocking=True)
            end.record(stream)
        end.synchronize()
        ms = start.elapsed_time(end)
        samples.append(ms * 1e6)
    samples.sort()
    med_ns = samples[len(samples) // 2]
    bw_gbs = size / med_ns
    return {
        "backend": "cuda_memcpy_async_d2h",
        "size_bytes": size,
        "lat_us_p50": round(med_ns / 1000, 3),
        "bw_GBs_p50": round(bw_gbs, 2),
        "iters": iters,
    }


def dsa_available() -> bool:
    return os.path.exists("/dev/dsa/wq0.0") or os.path.exists("/dev/dsa/wq1.0")


def bench_dsa_placeholder(size: int) -> dict:
    return {
        "backend": "dsa_descriptor_ring",
        "size_bytes": size,
        "lat_us_p50": None,
        "bw_GBs_p50": None,
        "note": "DSA WQ not enabled (accel-config enable-device blocked by harness)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/host_vllm_hybrid/lhc_phase1")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DSA char devices present: {dsa_available()}", flush=True)

    rows = []
    for label, size in SIZES:
        print(f"--- size={label} ({size} bytes) ---", flush=True)
        r = bench_glibc_memcpy(size)
        r["size_label"] = label
        rows.append(r); print(json.dumps(r), flush=True); gc.collect()
        r = bench_cuda_h2d(size)
        r["size_label"] = label
        rows.append(r); print(json.dumps(r), flush=True); gc.collect()
        r = bench_cuda_d2h(size)
        r["size_label"] = label
        rows.append(r); print(json.dumps(r), flush=True); gc.collect()
        r = bench_dsa_placeholder(size)
        r["size_label"] = label
        rows.append(r); print(json.dumps(r), flush=True)

    (out_dir / "dsa_memcpy_raw.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
