"""IDE_016 / TSK_025 — Latency microbench vs PyTorch baseline.

Measures fused sample (top-k + top-p + categorical) per-step latency.
Default: Qwen 2.5 shape (batch=32, vocab=152064), k=20, p=0.95, T=1.0.

Run::

    cd shadow_assists/features/IDE_016_avx512_amx_pool
    VLLM_USE_AVX512_SAMPLING=1 .venv/bin/python tests/bench_sampling_latency.py

CPU pinning (CLAUDE.md): 물리 코어만 사용 (0-111), HT 시블링 (112-223) 금지.
본 microbench 는 caller 측에서 `taskset -c 0-99 python ...` 으로 pin 권장.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_here, ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

os.environ.setdefault("VLLM_USE_AVX512_SAMPLING", "1")

import avx512_amx_pool as pool          # noqa: E402
from avx512_amx_pool import sampling    # noqa: E402


def bench_kernel(fn, *args, n_warmup=5, n_iter=100, **kwargs):
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for i in range(n_iter):
        fn(*args, rng_seed=i + 1, **kwargs) if "rng_seed" not in kwargs else fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000.0 / n_iter   # ms / iter


def bench_torch_topk_topp(logits, k, p, temperature, rng_seed=0):
    """PyTorch reference path equivalent to fused_sample."""
    return sampling._torch_fallback_fused_sample(
        logits, k=k, p=p, temperature=temperature, rng_seed=rng_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=152064)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    print("─" * 70)
    print(f"IDE_016 / TSK_025 — AVX-512 sampling latency microbench")
    print(f"  batch={args.batch}, vocab={args.vocab}, k={args.k}, "
          f"p={args.p}, T={args.temperature}, dtype={args.dtype}")
    print(f"  n_warmup={args.n_warmup}, n_iter={args.n_iter}")
    print(f"  CPU AVX-512: {sampling.cpu_has_avx512()}")
    print(f"  CPU AMX    : {pool.amx_is_available()}")
    print(f"  Kernel enabled: {pool.is_available()}")
    print("─" * 70)

    torch.manual_seed(42)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    logits = torch.randn(args.batch, args.vocab, dtype=dtype)

    # PyTorch baseline
    print("\n[1] PyTorch reference (CPU multinomial path):")
    t_torch = bench_kernel(
        bench_torch_topk_topp, logits, args.k, args.p, args.temperature,
        n_warmup=args.n_warmup, n_iter=args.n_iter)
    print(f"    {t_torch:8.3f} ms / iter")

    # AVX-512 fused
    if pool.is_available():
        print("\n[2] AVX-512 fused kernel:")
        # warm
        for _ in range(args.n_warmup):
            _ = sampling.fused_sample(logits, k=args.k, p=args.p,
                                     temperature=args.temperature, rng_seed=1)
        t0 = time.perf_counter()
        for i in range(args.n_iter):
            _ = sampling.fused_sample(logits, k=args.k, p=args.p,
                                     temperature=args.temperature, rng_seed=i + 1)
        t_avx = (time.perf_counter() - t0) * 1000.0 / args.n_iter
        print(f"    {t_avx:8.3f} ms / iter")
        speedup = t_torch / t_avx
        print(f"\n[Δ] speedup vs PyTorch reference: {speedup:.2f}x "
              f"({(speedup - 1) * 100:+.1f}%)")
    else:
        print("\n[2] AVX-512 kernel NOT enabled — set VLLM_USE_AVX512_SAMPLING=1")

    # Top-k only (component bench)
    print("\n[3] Component breakdown:")
    if pool.is_available():
        # top-k only
        for _ in range(args.n_warmup):
            _ = sampling.topk(logits, k=args.k)
        t0 = time.perf_counter()
        for i in range(args.n_iter):
            _ = sampling.topk(logits, k=args.k)
        t_topk = (time.perf_counter() - t0) * 1000.0 / args.n_iter
        print(f"    AVX-512 topk only      : {t_topk:8.3f} ms / iter")

        # softmax only (FP32 path)
        logits_f32 = logits.float()
        for _ in range(args.n_warmup):
            _ = sampling.softmax(logits_f32)
        t0 = time.perf_counter()
        for i in range(args.n_iter):
            _ = sampling.softmax(logits_f32)
        t_sm = (time.perf_counter() - t0) * 1000.0 / args.n_iter
        print(f"    AVX-512 softmax FP32   : {t_sm:8.3f} ms / iter")

    # PyTorch component
    t0 = time.perf_counter()
    for _ in range(args.n_iter):
        torch.topk(logits.float(), k=args.k, dim=-1)
    t_torch_topk = (time.perf_counter() - t0) * 1000.0 / args.n_iter
    print(f"    PyTorch topk            : {t_torch_topk:8.3f} ms / iter")
    t0 = time.perf_counter()
    for _ in range(args.n_iter):
        torch.softmax(logits.float(), dim=-1)
    t_torch_sm = (time.perf_counter() - t0) * 1000.0 / args.n_iter
    print(f"    PyTorch softmax         : {t_torch_sm:8.3f} ms / iter")

    print("─" * 70)


if __name__ == "__main__":
    main()
