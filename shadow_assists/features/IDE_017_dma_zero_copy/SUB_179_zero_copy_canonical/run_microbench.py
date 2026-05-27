#!/usr/bin/env python3
# SUB_179 — IDE_017 / TSK_029 zero-copy CPU compute path microbench
#
# Compares 3 GPU-read modes at fixed payload sizes:
#   (a) ZC   — zero-copy host-mapped buffer (CPU update → GPU kernel read via PCIe BAR)
#   (b) DMA  — cudaMemcpyAsync (CPU pinned src → device mirror → GPU kernel read)
#   (c) DEV  — device-only baseline (no host transfer, kernel cost lower bound)
#
# Each mode returns mean ms / iter as measured by cuda events on the same stream.
# Wall-time median is also recorded for cross-check.
#
# Output: zero_copy_microbench.json

from __future__ import annotations
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / "build" / "libzero_copy.so"


def load_lib():
    lib = ctypes.CDLL(str(LIB))
    lib.zc_check_device_support.restype = ctypes.c_int
    lib.zc_check_device_support.argtypes = [ctypes.c_int]

    lib.zc_bench_mode_zerocopy.restype = ctypes.c_double
    lib.zc_bench_mode_zerocopy.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int]

    lib.zc_bench_mode_dma.restype = ctypes.c_double
    lib.zc_bench_mode_dma.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int]

    lib.zc_bench_mode_devonly.restype = ctypes.c_double
    lib.zc_bench_mode_devonly.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int]

    lib.zc_bench_cpu_write.restype = ctypes.c_double
    lib.zc_bench_cpu_write.argtypes = [ctypes.c_size_t, ctypes.c_int]
    return lib


def main():
    lib = load_lib()

    gpu_id = int(os.environ.get("ZC_GPU_ID", "1"))  # avoid TP=0 conflict
    support = lib.zc_check_device_support(gpu_id)
    print(f"[SUB_179] GPU {gpu_id} canMapHostMemory = {support}")
    if support != 1:
        print("[SUB_179] ERROR: GPU does not support mapped host memory")
        sys.exit(2)

    # SUB_166 overhead-bound region: 4 KB ~ 256 KB (zero-copy candidate region)
    # plus crossover and bandwidth-bound region to verify SUB_166 finding.
    sizes_kb = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]
    # iters: small sizes need more iters for stable median
    iters_table = {
        4: 5000, 8: 5000, 16: 5000, 32: 2000, 64: 2000, 128: 1000, 256: 1000,
        512: 500, 1024: 500, 4096: 200, 16384: 100, 65536: 50,
    }

    out = {
        "sub_id": "SUB_179",
        "parent": "TSK_029",
        "scope": "zero-copy vs DMA microbench (4 KB ~ 64 MB)",
        "gpu_id": gpu_id,
        "results": [],
        "notes": [
            "Each row = mean ms/iter measured by cudaEvent on the same stream.",
            "ZC: cudaHostAllocMapped host buffer; GPU kernel reads via cudaHostGetDevicePointer.",
            "DMA: cudaHostAllocDefault pinned src; cudaMemcpyAsync H2D; GPU kernel reads from device buffer.",
            "DEV: cudaMalloc device src; kernel reads from device only (lower bound, no transfer).",
            "Kernel: copy_u32_kernel<<<32,256>>> — reads + writes n u32 (forces materialization).",
        ],
    }

    print(f"{'size_kb':>10} {'zc_us':>10} {'dma_us':>10} {'dev_us':>10} {'zc-dev_us':>12} {'dma-dev_us':>12} {'ratio_dma/zc':>14}")
    for kb in sizes_kb:
        bytes_ = kb * 1024
        iters = iters_table.get(kb, 100)

        t0 = time.perf_counter()
        zc_ms = lib.zc_bench_mode_zerocopy(bytes_, iters, gpu_id)
        t_zc = time.perf_counter() - t0

        t0 = time.perf_counter()
        dma_ms = lib.zc_bench_mode_dma(bytes_, iters, gpu_id)
        t_dma = time.perf_counter() - t0

        t0 = time.perf_counter()
        dev_ms = lib.zc_bench_mode_devonly(bytes_, iters, gpu_id)
        t_dev = time.perf_counter() - t0

        cpu_write_ms = lib.zc_bench_cpu_write(bytes_, max(100, iters // 10))

        zc_us = zc_ms * 1000.0
        dma_us = dma_ms * 1000.0
        dev_us = dev_ms * 1000.0
        ratio = dma_us / zc_us if zc_us > 0 else float("nan")

        print(f"{kb:>10} {zc_us:>10.2f} {dma_us:>10.2f} {dev_us:>10.2f} "
              f"{zc_us - dev_us:>12.2f} {dma_us - dev_us:>12.2f} {ratio:>14.3f}")

        out["results"].append({
            "size_kb": kb,
            "bytes": bytes_,
            "iters": iters,
            "zc_us_mean": round(zc_us, 3),
            "dma_us_mean": round(dma_us, 3),
            "devonly_us_mean": round(dev_us, 3),
            "cpu_write_us_mean": round(cpu_write_ms * 1000.0, 3),
            "zc_minus_dev_us": round(zc_us - dev_us, 3),
            "dma_minus_dev_us": round(dma_us - dev_us, 3),
            "ratio_dma_over_zc": round(ratio, 4),
            "wall_zc_s": round(t_zc, 3),
            "wall_dma_s": round(t_dma, 3),
            "wall_dev_s": round(t_dev, 3),
        })

    # crossover analysis
    crossover_kb = None
    for r in out["results"]:
        # ratio < 1 → DMA faster than ZC
        if r["ratio_dma_over_zc"] is not None and r["ratio_dma_over_zc"] < 1.0:
            crossover_kb = r["size_kb"]
            break
    out["crossover_kb_dma_beats_zc"] = crossover_kb

    out_path = HERE / "zero_copy_microbench.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SUB_179] wrote {out_path}")
    if crossover_kb is not None:
        print(f"[SUB_179] DMA beats zero-copy at size_kb >= {crossover_kb}")
    else:
        print(f"[SUB_179] zero-copy never beaten by DMA in measured range (4 KB ~ 64 MB)")


if __name__ == "__main__":
    main()
