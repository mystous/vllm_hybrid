#!/usr/bin/env python3
"""SUB_181 Step 2 — Jacobi cost vs acceptance trade-off analysis.

본 script:
1. SUB_180 microbench 결과 load
2. small-draft (Qwen 0.5B shape, hidden=896, vocab=152064) Jacobi LM-head
   microbench 실측 1-run + warmup 1
3. trade-off math:
     net_speedup(α, K, T_draft, T_verify) =
       E[accept] = (1 - α^(K+1)) / (1 - α)        # expected accepted tokens / cycle
       T_cycle    = T_draft + T_verify
       baseline   = 1 / T_verify (tokens/s without spec)
       spec_tps   = E[accept] / T_cycle
       speedup    = spec_tps / baseline
"""

import ctypes
import json
import os
import sys
import time
import numpy as np

LIB_PATH = "/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter/SUB_180_jacobi_canonical/build/libjacobi_avx512.so"
OUT_PATH = "/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter/SUB_181_jacobi_router_e2e/measurements/tradeoff_analysis.json"
SUB180_BENCH = "/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter/SUB_180_jacobi_canonical/jacobi_microbench_main.json"

# AGSD canonical reference: spec decode step ~30-50 ms (Qwen 32B TP4, ngram K=7)
GPU_VERIFY_MS_RANGE = (30.0, 50.0)
GPU_VERIFY_MS_TYPICAL = 40.0

# Acceptance rate estimates from SUB_180 §5
ACCEPT_RATE = {
    "chat":  {"min": 0.75, "max": 0.85, "typical": 0.80},
    "sonnet":{"min": 0.60, "max": 0.70, "typical": 0.65},
    "code":  {"min": 0.70, "max": 0.75, "typical": 0.72},
}


def fp32_to_bf16(arr_f32):
    a = arr_f32.astype(np.float32).view(np.uint32)
    bias = 0x7FFF + ((a >> 16) & 1)
    a = (a + bias) >> 16
    return a.astype(np.uint16)


def measure_lmhead_argmax(lib, hidden, vocab, BK, n_threads, iters=1, warmup=1):
    """Measure single LM-head argmax call (Jacobi inner-loop cost per iter)."""
    rng = np.random.default_rng(42)
    H = fp32_to_bf16(rng.standard_normal((BK, hidden), dtype=np.float32) * 0.1)
    W = fp32_to_bf16(rng.standard_normal((hidden, vocab), dtype=np.float32) * 0.02)
    H_ct = H.ctypes.data_as(ctypes.c_void_p)
    W_ct = W.ctypes.data_as(ctypes.c_void_p)
    argmax_out = np.zeros(BK, dtype=np.int32)
    maxlogit_out = np.zeros(BK, dtype=np.float32)
    arg_ct = argmax_out.ctypes.data_as(ctypes.c_void_p)
    mx_ct = maxlogit_out.ctypes.data_as(ctypes.c_void_p)
    for _ in range(warmup):
        lib.jacobi_lm_head_argmax_bf16(H_ct, W_ct, arg_ct, mx_ct,
                                       BK, hidden, vocab, n_threads)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        lib.jacobi_lm_head_argmax_bf16(H_ct, W_ct, arg_ct, mx_ct,
                                       BK, hidden, vocab, n_threads)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def setup_lib():
    lib = ctypes.CDLL(LIB_PATH)
    lib.jacobi_lm_head_argmax_bf16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.jacobi_lm_head_argmax_bf16.restype = None
    return lib


def expected_accept(alpha, K):
    """E[accepted tokens / spec cycle], including the 1 verified target token.
    Standard spec decoding result: E[n] = (1 - α^(K+1)) / (1 - α)."""
    if abs(1.0 - alpha) < 1e-9:
        return float(K + 1)
    return (1.0 - alpha**(K + 1)) / (1.0 - alpha)


def speedup(t_draft_ms, t_verify_ms, alpha, K):
    """spec_speedup = E[accept] * T_verify_baseline / T_cycle_spec.
    Here baseline = vanilla decode = T_verify (no draft). Spec cycle =
    T_draft + T_verify (single verify pass)."""
    e_accept = expected_accept(alpha, K)
    t_cycle = t_draft_ms + t_verify_ms
    # baseline produces 1 token / T_verify_ms; spec produces e_accept / t_cycle
    baseline_tps = 1.0 / t_verify_ms
    spec_tps = e_accept / t_cycle
    return spec_tps / baseline_tps


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    lib = setup_lib()

    # Load SUB_180 main bench (Qwen 32B target shape)
    with open(SUB180_BENCH) as f:
        sub180 = json.load(f)

    # Extract Qwen 32B (hidden=5120 vocab=152064) per-K best T=64 p50
    qwen32b_cost = {}  # K -> dict(BK, t_ms)
    if "results" in sub180:
        # find best p50 per K across B for T=64
        for r in sub180["results"]:
            K = r.get("K")
            B = r.get("B")
            T = r.get("threads") or r.get("T")
            p50 = r.get("p50_ms") or r.get("p50") or r.get("median_ms")
            if K is None or T is None or p50 is None:
                continue
            if T != 64:
                continue
            BK = K * B
            cur = qwen32b_cost.get(K)
            if cur is None or p50 < cur["t_ms"]:
                qwen32b_cost[K] = {"B": B, "BK": BK, "t_ms": p50}

    # Microbench small-draft shape (Qwen 0.5B: hidden=896 vocab=152064)
    print("[tradeoff] microbench small-draft Qwen 0.5B shape (hidden=896 vocab=152064)", flush=True)
    qwen05b_cost = {}
    for K in (3, 5, 7):
        for B in (1, 4):
            BK = K * B
            t_ms = measure_lmhead_argmax(lib, hidden=896, vocab=152064, BK=BK,
                                         n_threads=64, iters=1, warmup=1)
            print(f"  K={K} B={B} BK={BK} hidden=896 vocab=152064 T=64 p50={t_ms:.1f} ms",
                  flush=True)
            cur = qwen05b_cost.get(K)
            if cur is None or t_ms < cur["t_ms"]:
                qwen05b_cost[K] = {"B": B, "BK": BK, "t_ms": t_ms}

    # Also measure half-vocab partial top-N (vocab=8192) on Qwen 32B hidden
    print("[tradeoff] microbench partial-vocab (top-N=8192) Qwen 32B hidden", flush=True)
    partial_cost = {}
    for K in (5, 7):
        for B in (1, 4):
            BK = K * B
            t_ms = measure_lmhead_argmax(lib, hidden=5120, vocab=8192, BK=BK,
                                         n_threads=64, iters=1, warmup=1)
            print(f"  K={K} B={B} BK={BK} hidden=5120 vocab=8192 T=64 p50={t_ms:.1f} ms",
                  flush=True)
            cur = partial_cost.get(K)
            if cur is None or t_ms < cur["t_ms"]:
                partial_cost[K] = {"B": B, "BK": BK, "t_ms": t_ms}

    # Compute speedup matrix per workload × draft variant × K
    out = {
        "sub180_qwen32b_cost_ms": qwen32b_cost,
        "qwen05b_cost_ms": qwen05b_cost,
        "partial_vocab_cost_ms": partial_cost,
        "gpu_verify_ms_typical": GPU_VERIFY_MS_TYPICAL,
        "acceptance_rate_estimates": ACCEPT_RATE,
        "speedup_matrix": {},
    }
    drafts = {
        "qwen32b_full":  qwen32b_cost,
        "qwen05b_small": qwen05b_cost,
        "partial_top8k": partial_cost,
    }
    for draft_name, cost_map in drafts.items():
        out["speedup_matrix"][draft_name] = {}
        for K, info in sorted(cost_map.items()):
            t_draft = info["t_ms"]
            out["speedup_matrix"][draft_name][f"K={K}"] = {
                "t_draft_ms": t_draft,
                "B_at_best": info["B"],
                "BK": info["BK"],
                "verify_ms": GPU_VERIFY_MS_TYPICAL,
                "by_workload": {},
            }
            for wl, ar in ACCEPT_RATE.items():
                row = {}
                for tag in ("min", "typical", "max"):
                    alpha = ar[tag]
                    sp = speedup(t_draft, GPU_VERIFY_MS_TYPICAL, alpha, K)
                    row[f"alpha={alpha:.2f}"] = round(sp, 4)
                out["speedup_matrix"][draft_name][f"K={K}"]["by_workload"][wl] = row

    # Net-win flag
    out["any_net_win"] = False
    out["net_win_cases"] = []
    for draft_name, by_K in out["speedup_matrix"].items():
        for Ktag, info in by_K.items():
            for wl, by_alpha in info["by_workload"].items():
                for atag, sp in by_alpha.items():
                    if sp > 1.05:
                        out["any_net_win"] = True
                        out["net_win_cases"].append({
                            "draft": draft_name, "K": Ktag, "workload": wl,
                            "alpha_case": atag, "speedup": sp,
                        })

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[tradeoff] wrote {OUT_PATH}", flush=True)
    print(f"[tradeoff] any_net_win={out['any_net_win']} cases={len(out['net_win_cases'])}",
          flush=True)
    if out["net_win_cases"]:
        for c in out["net_win_cases"][:6]:
            print(f"  - {c}", flush=True)


if __name__ == "__main__":
    main()
