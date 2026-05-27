"""SUB_175 — AMX BF16 matmul microbench (prod Sapphire Rapids 8480+).

설계 변경: AMX OP 은 single-thread (calling thread, has permission) only.
PT BF16 matmul 은 별도 측정. 동일 process 에서 OMP fork 시 backround
thread 가 AMX permission 없어 SIGILL 가능 — 따라서:

  - run-1: AMX only, 1 thread (taskset 0)
  - run-2: PT BF16 multi-thread (taskset 0-3, OMP=4) — separate process

shapes:
  - Qwen 7B: hidden=3584, intermediate=18944 (B=128, 256)
  - Qwen 32B: hidden=5120, intermediate=27648 (B=64, 128, 256)
  - representative: [B, K] · [K, N] → [B, N]
"""
from __future__ import annotations

import argparse, json, os, sys, time

# Limit BLAS pool to avoid SIGILL when AMX is mixed with OMP fork
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

sys.path.insert(0, "/workspace/vllm_hybrid/shadow_assists/features/IDE_016_avx512_amx_pool")
import avx512_amx_pool

torch.set_num_threads(int(os.environ.get("PT_THREADS", "1")))


def bench(label, fn, warmup=2, iters=5):
    """min over `iters` runs (after warmup), returns seconds-per-iter (best)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def tflops(M, K, N, sec):
    return 2.0 * M * K * N / sec / 1e12


SHAPES = {
    # name: (K, N)
    "qwen7b_gate_up":  (3584, 18944),
    "qwen7b_down":     (18944, 3584),
    "qwen32b_gate_up": (5120, 27648),
    "qwen32b_down":    (27648, 5120),
}

BATCHES = [64, 128, 256]


def run_amx():
    """AMX only — single-thread (taskset 0)."""
    avx512_amx_pool.matmul.request_amx_permission()
    print(f"AMX avail: {avx512_amx_pool.amx_is_available()}, "
          f"perm granted true, threads={torch.get_num_threads()}")
    rows = []
    for shape_name, (K, N) in SHAPES.items():
        for B in BATCHES:
            A = torch.randn(B, K, dtype=torch.bfloat16)
            Bmat = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
            B_packed = avx512_amx_pool.matmul.amx_repack_b(Bmat)

            def amx_run():
                return avx512_amx_pool.matmul.amx_matmul(A, B_packed)

            amx_best, amx_avg = bench("amx", amx_run, warmup=2, iters=5)
            amx_tf = tflops(B, K, N, amx_best)
            rows.append({
                "shape": shape_name, "B": B, "K": K, "N": N,
                "amx_best_ms": amx_best * 1e3,
                "amx_avg_ms":  amx_avg  * 1e3,
                "amx_tflops":  amx_tf,
            })
            print(f"  AMX {shape_name:18s} B={B:4d} K={K:5d} N={N:5d} "
                  f"best={amx_best*1e3:7.2f}ms avg={amx_avg*1e3:7.2f}ms "
                  f"tflops={amx_tf:6.3f}")

    # correctness check on small shape
    A = torch.randn(16, 32, dtype=torch.bfloat16)
    Bm = torch.randn(32, 16, dtype=torch.bfloat16)
    Bp = avx512_amx_pool.matmul.amx_repack_b(Bm)
    C = avx512_amx_pool.matmul.amx_matmul(A, Bp).float()
    ref = (A.float() @ Bm.float())
    err = (C - ref).abs().max().item()
    print(f"  [correctness] M=16 K=32 N=16 max_abs_err={err:.4e}")

    with open("/tmp/sub175_amx_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_pt(threads):
    torch.set_num_threads(threads)
    print(f"PT BF16 (threads={threads}, set_num_threads={torch.get_num_threads()})")
    rows = []
    for shape_name, (K, N) in SHAPES.items():
        for B in BATCHES:
            A = torch.randn(B, K, dtype=torch.bfloat16)
            Bmat = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

            def pt_run():
                return torch.matmul(A, Bmat)

            pt_best, pt_avg = bench("pt", pt_run, warmup=2, iters=5)
            pt_tf = tflops(B, K, N, pt_best)
            rows.append({
                "shape": shape_name, "B": B, "K": K, "N": N,
                "pt_threads": threads,
                "pt_best_ms": pt_best * 1e3,
                "pt_avg_ms":  pt_avg  * 1e3,
                "pt_tflops":  pt_tf,
            })
            print(f"  PT  {shape_name:18s} B={B:4d} K={K:5d} N={N:5d} "
                  f"best={pt_best*1e3:7.2f}ms avg={pt_avg*1e3:7.2f}ms "
                  f"tflops={pt_tf:6.3f}")
    with open(f"/tmp/sub175_pt_rows_t{threads}.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["amx", "pt1", "pt4", "pt32"])
    args = ap.parse_args()
    if args.mode == "amx":
        run_amx()
    elif args.mode == "pt1":
        run_pt(1)
    elif args.mode == "pt4":
        run_pt(4)
    elif args.mode == "pt32":
        run_pt(32)
