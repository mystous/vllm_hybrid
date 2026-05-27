#!/usr/bin/env python3
"""SUB_185 ON-mode CPU dequant firer.

Continuously calls SUB_178 `cold_kv_int8_to_bf16` AVX-512 kernel at the
target rate (Hz) while the long-context benchmark is running. This is a
PROXY for "cold-KV CPU dequant overlapped with GPU prefill" — the actual
dequant output is discarded (not injected into vLLM KV cache).

Goal: measure whether concurrent CPU-side dequant
  (a) slows down GPU prefill via shared resources (memory bus, PCIe), or
  (b) is truly overlap-friendly (orthogonal to GPU compute).

Exits cleanly on SIGTERM (sets running=False).
Writes per-iteration tally to <out>.json.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

LIB_PATH = Path(
    "/workspace/vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/"
    "SUB_178_cold_kv_decompress/build/libcold_kv.so"
)


def _load_lib():
    lib = C.CDLL(str(LIB_PATH))
    # void cold_kv_int8_to_bf16(const int8_t* q, const uint16_t* scale_bf16,
    #                           uint16_t* out_bf16, int n_elems, int scale_group_size)
    lib.cold_kv_int8_to_bf16.argtypes = [
        C.POINTER(C.c_int8), C.POINTER(C.c_uint16),
        C.POINTER(C.c_uint16), C.c_int, C.c_int,
    ]
    lib.cold_kv_int8_to_bf16.restype = None
    return lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-elems", type=int, default=262144,
                    help="Elements per call (default 256 KB ~ SUB_178 sweet spot).")
    ap.add_argument("--scale-group", type=int, default=128)
    ap.add_argument("--target-hz", type=float, default=100.0,
                    help="Target call rate. 0 = as fast as possible.")
    ap.add_argument("--duration-s", type=float, default=600.0,
                    help="Max duration (firer killed externally before).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lib = _load_lib()

    n = args.n_elems
    g = args.scale_group
    n_groups = max(1, (n + g - 1) // g)

    rng = np.random.default_rng(0)
    q = rng.integers(-100, 100, size=n, dtype=np.int8)
    # bf16 stored as uint16 (high 16 bits of fp32). Pre-fill with ~0.1 scale.
    s32 = np.full(n_groups, 0.1, dtype=np.float32)
    scale_u16 = (s32.view(np.uint32) >> 16).astype(np.uint16)
    out = np.zeros(n, dtype=np.uint16)

    q_p = q.ctypes.data_as(C.POINTER(C.c_int8))
    s_p = scale_u16.ctypes.data_as(C.POINTER(C.c_uint16))
    o_p = out.ctypes.data_as(C.POINTER(C.c_uint16))

    running = {"v": True}

    def _stop(*_args):
        running["v"] = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    period = 1.0 / args.target_hz if args.target_hz > 0 else 0.0
    t0 = time.perf_counter()
    next_t = t0
    iters = 0
    total_elems = 0
    while running["v"] and (time.perf_counter() - t0) < args.duration_s:
        lib.cold_kv_int8_to_bf16(q_p, s_p, o_p, n, g)
        iters += 1
        total_elems += n
        if period > 0:
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # behind schedule — reset next_t so we don't busy-loop
                next_t = time.perf_counter()

    wall = time.perf_counter() - t0
    eff_hz = iters / wall if wall > 0 else 0.0
    eff_bw_gbs = (total_elems / wall) / 1e9 if wall > 0 else 0.0
    args.out.write_text(json.dumps({
        "iters": iters,
        "n_elems_per_call": n,
        "scale_group": g,
        "target_hz": args.target_hz,
        "effective_hz": eff_hz,
        "wall_s": wall,
        "effective_bw_GB_s_int8_in": eff_bw_gbs,
    }, indent=2))


if __name__ == "__main__":
    main()
