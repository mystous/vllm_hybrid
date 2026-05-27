#!/usr/bin/env python3
"""SUB_187 AMX draft head microbench — per-step latency for Qwen 0.5B small draft.

Measures LM-head matmul (BF16, hidden=896 vocab=152064) on AMX, which is the
dominant cost in SUB_181 Jacobi K=7 245 ms breakdown. Target: < 5 ms / step.
"""
import argparse
import ctypes
import os
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "build" / "libamx_draft_qwen05b.so"


def load_lib():
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.amx_draft_qwen05b_init.restype = ctypes.c_int
    lib.amx_draft_qwen05b_init.argtypes = []
    lib.amx_draft_qwen05b_free.restype = None
    lib.amx_draft_qwen05b_free.argtypes = []
    lib.amx_draft_qwen05b_step_ms.restype = ctypes.c_double
    lib.amx_draft_qwen05b_step_ms.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.amx_draft_qwen05b_single_ms.restype = ctypes.c_double
    lib.amx_draft_qwen05b_single_ms.argtypes = [ctypes.c_int]
    lib.amx_draft_qwen05b_mlp_ms.restype = ctypes.c_double
    lib.amx_draft_qwen05b_mlp_ms.argtypes = [ctypes.c_int]
    lib.amx_draft_qwen05b_hw_amx.restype = ctypes.c_int
    lib.amx_draft_qwen05b_hw_amx.argtypes = []
    return lib


def bench(lib, B, K, n_iter=20, warmup=3):
    samples = []
    # Warmup
    for _ in range(warmup):
        lib.amx_draft_qwen05b_step_ms(B, K)
    for _ in range(n_iter):
        ms = lib.amx_draft_qwen05b_step_ms(B, K)
        samples.append(ms)
    samples.sort()
    return {
        "B": B, "K": K,
        "p50_ms": samples[len(samples) // 2],
        "p90_ms": samples[int(len(samples) * 0.9)],
        "p99_ms": samples[-1],
        "mean_ms": statistics.mean(samples),
        "n": n_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=20)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    args = ap.parse_args()

    lib = load_lib()
    has_amx = lib.amx_draft_qwen05b_hw_amx()
    print(f"[hw] AMX available = {bool(has_amx)}")
    print(f"[hw] OMP threads (env) = {args.threads}")

    rc = lib.amx_draft_qwen05b_init()
    if rc != 0:
        print(f"ERROR: init returned {rc}")
        return 1
    try:
        # First: single LM-head matmul cost at varying B
        print("\n=== Single LM-head matmul (1 step) ===")
        for B in (1, 4, 16):
            r = bench(lib, B, 1, n_iter=args.iter)
            print(f"B={B:2d} K=1 : p50={r['p50_ms']:.3f} ms p90={r['p90_ms']:.3f} mean={r['mean_ms']:.3f}")

        # Second: K=5 / K=7 draft loop (cumulative AMX matmul chain)
        print("\n=== K-step draft loop (LM-head × K) ===")
        results = []
        for B in (1, 4):
            for K in (5, 7):
                r = bench(lib, B, K, n_iter=args.iter)
                r["per_step_ms"] = r["p50_ms"] / K
                results.append(r)
                print(f"B={B:2d} K={K} : total p50={r['p50_ms']:.3f} ms  "
                      f"per-step={r['per_step_ms']:.3f} ms  "
                      f"mean={r['mean_ms']:.3f} ms")

        # MLP timing (per-layer linear cost)
        print("\n=== Per-layer MLP-gate matmul (hidden × intermediate) ===")
        for B in (1, 4, 16):
            ms = lib.amx_draft_qwen05b_mlp_ms(B)
            print(f"B={B:2d} MLP-gate ms = {ms:.3f}")

        # Verdict
        print("\n=== Verdict ===")
        target_step_ms = 5.0
        for r in results:
            verdict = "PASS" if r["per_step_ms"] < target_step_ms else "FAIL"
            print(f"  B={r['B']} K={r['K']}: per-step {r['per_step_ms']:.2f} ms "
                  f"(target < {target_step_ms} ms) → {verdict}")

    finally:
        lib.amx_draft_qwen05b_free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
