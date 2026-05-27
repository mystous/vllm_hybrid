#!/usr/bin/env python3
"""
SUB_180 — Jacobi AVX-512 LM-head argmax microbench

Measures the BF16 LM-head argmax kernel that powers Jacobi parallel decoding's
inner loop. The Jacobi driver re-uses this kernel per fixed-point iteration;
this microbench focuses on per-iteration cost.

config: Qwen 32B target shape (hidden=5120, vocab=152064)
sweeps: K in {3, 5, 7, 9}, B in {1, 4, 8}
"""

import argparse
import ctypes
import json
import os
import sys
import time
import numpy as np

# ----- env / args -----
parser = argparse.ArgumentParser()
parser.add_argument("--lib", default=os.path.join(os.path.dirname(__file__), "build", "libjacobi_avx512.so"))
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "jacobi_microbench.json"))
parser.add_argument("--hidden", type=int, default=5120)
parser.add_argument("--vocab", type=int, default=152064)
parser.add_argument("--K_list", default="3,5,7,9")
parser.add_argument("--B_list", default="1,4,8")
parser.add_argument("--threads", default="1,8,16")
parser.add_argument("--iters", type=int, default=3)  # measurement reps (1 run rule -> set to 1 below)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--verify", action="store_true", help="compare vs scalar ref (small shape)")
parser.add_argument("--verify_hidden", type=int, default=128)
parser.add_argument("--verify_vocab", type=int, default=512)
args = parser.parse_args()

print(f"[SUB_180] lib={args.lib} hidden={args.hidden} vocab={args.vocab}", flush=True)

# load .so
lib = ctypes.CDLL(args.lib)
lib.jacobi_lm_head_argmax_bf16.argtypes = [
    ctypes.c_void_p,  # H
    ctypes.c_void_p,  # W
    ctypes.c_void_p,  # argmax_out
    ctypes.c_void_p,  # maxlogit_out
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.jacobi_lm_head_argmax_bf16.restype = None

lib.jacobi_lm_head_argmax_scalar_ref.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.jacobi_lm_head_argmax_scalar_ref.restype = None

lib.jacobi_run.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p,
]
lib.jacobi_run.restype = ctypes.c_int

# ----- bf16 helpers -----
def fp32_to_bf16(arr_f32):
    # Round to nearest even by adding 0x7FFF + LSB-of-top16
    a = arr_f32.astype(np.float32).view(np.uint32)
    bias = 0x7FFF + ((a >> 16) & 1)
    a = (a + bias) >> 16
    return a.astype(np.uint16)

def bf16_to_fp32(arr_u16):
    a = arr_u16.astype(np.uint32) << 16
    return a.view(np.float32)

# ----- accuracy check on small shape -----
def verify_accuracy():
    hidden = args.verify_hidden
    vocab  = args.verify_vocab
    BK = 4
    np.random.seed(42)
    # keep values small so BF16 rounding doesn't dominate
    H_f = (np.random.randn(BK, hidden) * 0.1).astype(np.float32)
    W_f = (np.random.randn(hidden, vocab) * 0.1).astype(np.float32)
    H_b = fp32_to_bf16(H_f).reshape(-1)
    W_b = fp32_to_bf16(W_f).reshape(-1)

    out_v = np.zeros(BK, dtype=np.int32)
    out_s = np.zeros(BK, dtype=np.int32)
    lib.jacobi_lm_head_argmax_bf16(
        H_b.ctypes.data, W_b.ctypes.data,
        out_v.ctypes.data, 0, BK, hidden, vocab, 1)
    lib.jacobi_lm_head_argmax_scalar_ref(
        H_b.ctypes.data, W_b.ctypes.data,
        out_s.ctypes.data, 0, BK, hidden, vocab)

    # also numpy ref using bf16->fp32 conversion (same precision domain)
    H_q = bf16_to_fp32(H_b).reshape(BK, hidden)
    W_q = bf16_to_fp32(W_b).reshape(hidden, vocab)
    logits = H_q @ W_q
    np_ref = np.argmax(logits, axis=1).astype(np.int32)

    return {
        "BK": BK, "hidden": hidden, "vocab": vocab,
        "avx_vs_scalar_match": int((out_v == out_s).sum()),
        "avx_vs_numpy_match":  int((out_v == np_ref).sum()),
        "scalar_vs_numpy_match": int((out_s == np_ref).sum()),
        "out_avx": out_v.tolist(),
        "out_scalar": out_s.tolist(),
        "out_numpy": np_ref.tolist(),
    }

if args.verify:
    accuracy = verify_accuracy()
    print(f"[SUB_180] verify: {accuracy}", flush=True)
else:
    accuracy = None

# ----- bench shapes -----
K_list = [int(x) for x in args.K_list.split(",")]
B_list = [int(x) for x in args.B_list.split(",")]
T_list = [int(x) for x in args.threads.split(",")]

hidden = args.hidden
vocab  = args.vocab

# Build shared W once (large: hidden*vocab*2 = 5120*152064*2 = ~1.56 GB!).
# To keep memory tractable for the microbench we reduce vocab if requested.
print(f"[SUB_180] allocating W [hidden={hidden}, vocab={vocab}] BF16 -> ~{hidden*vocab*2/1e9:.2f} GB", flush=True)
W_f = (np.random.RandomState(7).randn(hidden, vocab) * 0.02).astype(np.float32)
W_b = fp32_to_bf16(W_f).reshape(-1)
print(f"[SUB_180] W ready (np buffer {W_b.nbytes/1e9:.2f} GB)", flush=True)

cells = []

for K in K_list:
    for B in B_list:
        BK = B * K
        # H per (B,K) row
        H_f = (np.random.RandomState(13 + B*100 + K).randn(BK, hidden) * 0.1).astype(np.float32)
        H_b = fp32_to_bf16(H_f).reshape(-1)
        out = np.zeros(BK, dtype=np.int32)

        for T in T_list:
            # warmup
            for _ in range(args.warmup):
                lib.jacobi_lm_head_argmax_bf16(
                    H_b.ctypes.data, W_b.ctypes.data,
                    out.ctypes.data, 0, BK, hidden, vocab, T)
            # measure
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                lib.jacobi_lm_head_argmax_bf16(
                    H_b.ctypes.data, W_b.ctypes.data,
                    out.ctypes.data, 0, BK, hidden, vocab, T)
                t1 = time.perf_counter()
                samples.append((t1 - t0) * 1000.0)  # ms
            samples.sort()
            p50 = samples[len(samples)//2]
            mean = sum(samples) / len(samples)
            flops_per_call = 2.0 * BK * hidden * vocab
            tflops = flops_per_call / (p50 / 1000.0) / 1e12
            cell = {
                "K": K, "B": B, "BK": BK, "threads": T,
                "p50_ms": p50, "mean_ms": mean, "samples_ms": samples,
                "tflops_p50": tflops,
                "argmax_first": int(out[0]),
            }
            print(f"[SUB_180] K={K:>2} B={B:>2} BK={BK:>3} T={T:>3} p50={p50:>9.3f}ms mean={mean:>9.3f}ms tflops={tflops:.4f}", flush=True)
            cells.append(cell)

result = {
    "config": {
        "lib": args.lib,
        "hidden": hidden, "vocab": vocab,
        "K_list": K_list, "B_list": B_list, "threads_list": T_list,
        "iters": args.iters, "warmup": args.warmup,
    },
    "accuracy": accuracy,
    "cells": cells,
}

with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(f"[SUB_180] saved {args.out}", flush=True)
