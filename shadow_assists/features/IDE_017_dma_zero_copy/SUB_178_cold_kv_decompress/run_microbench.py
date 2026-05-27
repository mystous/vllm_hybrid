#!/usr/bin/env python3
"""SUB_178 — Cold-KV CPU dequant microbench.

Measures:
  - AVX-512 INT8/INT4 → BF16 dequant kernel throughput (GB/s, GFLOPS-equiv)
  - Accuracy vs reference scalar implementation (max abs diff)
  - Optional: dequant + DMA-push pipeline overhead (uses SUB_176 pinned_pool)

Single-run per measurement (user 1-run rule). Block sizes chosen to match
Qwen 32B TP=4 KV layout: 16 (block_size) × 2 (heads/rank) × 128 (head_dim) =
4096 elems / chunk → 8 KB BF16, 4 KB INT8, 2 KB INT4. Plus larger sweep.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "build" / "libcold_kv.so"

assert LIB_PATH.exists(), f"libcold_kv.so missing at {LIB_PATH}; build first"

lib = ctypes.CDLL(str(LIB_PATH))

# void cold_kv_int8_to_bf16(int8_t* q, uint16_t* scale, uint16_t* out, int n, int gs)
lib.cold_kv_int8_to_bf16.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
]
lib.cold_kv_int8_to_bf16.restype = None

lib.cold_kv_int4_to_bf16.argtypes = lib.cold_kv_int8_to_bf16.argtypes
lib.cold_kv_int4_to_bf16.restype = None

lib.cold_kv_int8_to_bf16_ref.argtypes = lib.cold_kv_int8_to_bf16.argtypes
lib.cold_kv_int8_to_bf16_ref.restype = None

lib.cold_kv_int4_to_bf16_ref.argtypes = lib.cold_kv_int8_to_bf16.argtypes
lib.cold_kv_int4_to_bf16_ref.restype = None


def bf16_view_uint16(arr_bf16: np.ndarray) -> np.ndarray:
    """View ml_dtypes.bfloat16 array as uint16 (raw bits)."""
    return arr_bf16.view(np.uint16)


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Convert float32 array → bf16 raw uint16 bits (truncate, RTNE not needed
    for scale generation)."""
    u32 = x.astype(np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def time_kernel(fn, n_iters: int) -> float:
    """Returns median per-call elapsed (sec)."""
    samples = []
    # warm up
    for _ in range(5):
        fn()
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_int8(n_elems: int, gs: int, n_iters: int) -> dict:
    rng = np.random.default_rng(42)
    q = rng.integers(-127, 128, size=n_elems, dtype=np.int8)
    n_groups = (n_elems + gs - 1) // gs
    scale_f32 = (rng.random(n_groups, dtype=np.float32) * 0.1 + 0.01).astype(np.float32)
    scale_bf16 = fp32_to_bf16_bits(scale_f32)
    out = np.zeros(n_elems, dtype=np.uint16)
    out_ref = np.zeros(n_elems, dtype=np.uint16)

    q_p = q.ctypes.data_as(ctypes.c_void_p)
    s_p = scale_bf16.ctypes.data_as(ctypes.c_void_p)
    o_p = out.ctypes.data_as(ctypes.c_void_p)
    or_p = out_ref.ctypes.data_as(ctypes.c_void_p)

    # accuracy: compare AVX-512 vs scalar reference
    lib.cold_kv_int8_to_bf16(q_p, s_p, o_p, n_elems, gs)
    lib.cold_kv_int8_to_bf16_ref(q_p, s_p, or_p, n_elems, gs)
    avx_fp32 = bf16_bits_to_fp32(out)
    ref_fp32 = bf16_bits_to_fp32(out_ref)
    max_abs = float(np.max(np.abs(avx_fp32 - ref_fp32)))
    # Quant induced error vs true (q * scale)
    true_fp32 = q.astype(np.float32) * np.repeat(scale_f32, gs)[:n_elems]
    quant_max_abs = float(np.max(np.abs(ref_fp32 - true_fp32)))

    fn = lambda: lib.cold_kv_int8_to_bf16(q_p, s_p, o_p, n_elems, gs)
    med = time_kernel(fn, n_iters)
    # Throughput: input bytes (INT8) + output bytes (BF16) per sec
    in_bytes = n_elems  # int8
    out_bytes = n_elems * 2  # bf16
    bw_gbs = (in_bytes + out_bytes) / med / 1e9
    return {
        "n_elems": n_elems,
        "group_size": gs,
        "iters": n_iters,
        "median_us": med * 1e6,
        "throughput_gbs": bw_gbs,
        "elems_per_sec": n_elems / med,
        "max_abs_avx_vs_ref": max_abs,
        "quant_max_abs_vs_true": quant_max_abs,
    }


def bench_int4(n_elems: int, gs: int, n_iters: int) -> dict:
    rng = np.random.default_rng(43)
    # generate signed int4 in [-8, 7], pack 2 per byte (low nibble first)
    nib = rng.integers(-8, 8, size=n_elems, dtype=np.int8)
    # pack
    assert n_elems % 2 == 0, "n_elems must be even for INT4 packing"
    packed = np.zeros(n_elems // 2, dtype=np.uint8)
    nib_u = (nib & 0x0F).astype(np.uint8)
    packed[:] = nib_u[0::2] | (nib_u[1::2] << 4)

    n_groups = (n_elems + gs - 1) // gs
    scale_f32 = (rng.random(n_groups, dtype=np.float32) * 0.1 + 0.01).astype(np.float32)
    scale_bf16 = fp32_to_bf16_bits(scale_f32)
    out = np.zeros(n_elems, dtype=np.uint16)
    out_ref = np.zeros(n_elems, dtype=np.uint16)

    q_p = packed.ctypes.data_as(ctypes.c_void_p)
    s_p = scale_bf16.ctypes.data_as(ctypes.c_void_p)
    o_p = out.ctypes.data_as(ctypes.c_void_p)
    or_p = out_ref.ctypes.data_as(ctypes.c_void_p)

    lib.cold_kv_int4_to_bf16(q_p, s_p, o_p, n_elems, gs)
    lib.cold_kv_int4_to_bf16_ref(q_p, s_p, or_p, n_elems, gs)
    avx_fp32 = bf16_bits_to_fp32(out)
    ref_fp32 = bf16_bits_to_fp32(out_ref)
    max_abs = float(np.max(np.abs(avx_fp32 - ref_fp32)))
    true_fp32 = nib.astype(np.float32) * np.repeat(scale_f32, gs)[:n_elems]
    quant_max_abs = float(np.max(np.abs(ref_fp32 - true_fp32)))

    fn = lambda: lib.cold_kv_int4_to_bf16(q_p, s_p, o_p, n_elems, gs)
    med = time_kernel(fn, n_iters)
    in_bytes = n_elems // 2  # int4 packed
    out_bytes = n_elems * 2
    bw_gbs = (in_bytes + out_bytes) / med / 1e9
    return {
        "n_elems": n_elems,
        "group_size": gs,
        "iters": n_iters,
        "median_us": med * 1e6,
        "throughput_gbs": bw_gbs,
        "elems_per_sec": n_elems / med,
        "max_abs_avx_vs_ref": max_abs,
        "quant_max_abs_vs_true": quant_max_abs,
    }


def main() -> int:
    # Qwen 32B TP=4 KV chunk = 16 * 2 * 128 = 4096 elems
    # We also include multi-chunk sizes representative of cold-region batches:
    #   1 chunk = 4096, 8 chunks = 32768, 64 chunks = 262144,
    #   512 chunks = 2097152, 4096 chunks = 16777216 (~32 MB BF16)
    sizes = [4096, 32768, 262144, 2097152, 16777216]

    results = {"int8": [], "int4": []}
    for n in sizes:
        # per-head scale: group_size = 128 (head_dim) matches realistic quant
        gs = 128
        n_iters = 200 if n <= 262144 else (50 if n <= 2097152 else 10)
        r8 = bench_int8(n, gs, n_iters)
        r4 = bench_int4(n, gs, n_iters)
        results["int8"].append(r8)
        results["int4"].append(r4)
        print(f"[INT8] n={n:>9} gs={gs:>3} med={r8['median_us']:>10.2f} us  "
              f"BW={r8['throughput_gbs']:>6.2f} GB/s  "
              f"acc(avx-ref)={r8['max_abs_avx_vs_ref']:.2e}  "
              f"acc(ref-true)={r8['quant_max_abs_vs_true']:.2e}")
        print(f"[INT4] n={n:>9} gs={gs:>3} med={r4['median_us']:>10.2f} us  "
              f"BW={r4['throughput_gbs']:>6.2f} GB/s  "
              f"acc(avx-ref)={r4['max_abs_avx_vs_ref']:.2e}  "
              f"acc(ref-true)={r4['quant_max_abs_vs_true']:.2e}")

    out_path = HERE / "cold_kv_microbench.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
