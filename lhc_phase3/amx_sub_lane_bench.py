"""
LHC Phase 3 — AMX sub-lane re-selection microbench.

Phase 1 (sampler softmax/topk) and Phase 2 (logits-head matmul) both FAIL.
We measure 5 new candidate sub-lanes:

  C1 draft head    : hidden[bs, 2048] @ embed_T[2048, vocab]  (smaller hidden)
  C2 RMS norm      : x * rsqrt(mean(x^2)+eps) * weight  (memory-bound)
  C3 prefix scan   : byte-level compare + hash (radix-tree lookup)
  C4 KV scale calib: per-layer per-head bf16 abs-max reduce
  C5 fused norm+add: (x+res) -> rmsnorm  (2 element-wise + reduce)

Backends:
  amx     : torch CPU bf16 (oneDNN default ISA → avx10_1_512_amx on EMR)
  avx512  : child re-exec w/ ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16
  gpu     : torch CUDA bf16 (single B200, sm_100)

Gate (per sub-lane):
  AMX latency ≤ 3× GPU  OR  CPU stall ≈ 0% (ortho-lane)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# common runner
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS = 30


def _measure(fn, is_cuda: bool):
    for _ in range(WARMUP):
        fn()
        if is_cuda:
            torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)  # μs
    samples.sort()
    p50 = samples[len(samples) // 2]
    p10 = samples[int(len(samples) * 0.10)]
    p90 = samples[int(len(samples) * 0.90)]
    return {"p10_us": p10, "p50_us": p50, "p90_us": p90}


# ---------------------------------------------------------------------------
# C1 — draft head matmul: small hidden (2048), still big vocab
# ---------------------------------------------------------------------------

C1_CONFIGS = [
    # (name, vocab, hidden)
    ("qwen-152k_h2048", 152_064, 2048),
    ("llama-128k_h2048", 128_256, 2048),
    ("draft_h2048_v32k", 32_000, 2048),  # smaller draft vocab
]
C1_BATCHES = [1, 4, 16, 64]


def bench_c1(device: str, dtype: torch.dtype):
    out = []
    is_cuda = device.startswith("cuda")
    for name, vocab, hidden in C1_CONFIGS:
        for bs in C1_BATCHES:
            torch.manual_seed(0)
            h = torch.randn(bs, hidden, dtype=torch.float32).to(device=device, dtype=dtype)
            w = torch.randn(hidden, vocab, dtype=torch.float32).to(device=device, dtype=dtype)

            def step(h=h, w=w):
                return torch.matmul(h, w)

            r = _measure(step, is_cuda)
            r["config"] = f"{name}_bs{bs}"
            r["flops"] = 2 * bs * hidden * vocab
            r["tflops"] = r["flops"] / (r["p50_us"] * 1e-6) / 1e12
            out.append(r)
            print(f"[C1][{device}] {r['config']:32s} p50={r['p50_us']:9.1f} us  {r['tflops']:6.2f} TF", flush=True)
    return out


# ---------------------------------------------------------------------------
# C2 — RMS norm (memory-bound)
# ---------------------------------------------------------------------------

# (bs*seq_len, hidden). seq_len effective during prefill batching.
C2_CONFIGS = [
    ("tok2k_h4096", 2048, 4096),
    ("tok8k_h4096", 8192, 4096),
    ("tok2k_h8192", 2048, 8192),
    ("tok8k_h8192", 8192, 8192),
]


def bench_c2(device: str, dtype: torch.dtype):
    out = []
    is_cuda = device.startswith("cuda")
    eps = 1e-6
    for name, tok, hidden in C2_CONFIGS:
        torch.manual_seed(0)
        x = torch.randn(tok, hidden, dtype=torch.float32).to(device=device, dtype=dtype)
        w = torch.randn(hidden, dtype=torch.float32).to(device=device, dtype=dtype)

        def step(x=x, w=w):
            # FP32-promoted rmsnorm (vLLM canonical form)
            xf = x.to(torch.float32)
            v = xf.pow(2).mean(-1, keepdim=True)
            xn = xf * torch.rsqrt(v + eps)
            return (xn.to(dtype)) * w

        r = _measure(step, is_cuda)
        r["config"] = f"{name}"
        # bytes touched: read x bf16 (2B), read w bf16, write y bf16 ≈ 3 × tok × hidden × 2
        r["bytes"] = 3 * tok * hidden * 2
        r["gbps"] = r["bytes"] / (r["p50_us"] * 1e-6) / 1e9
        out.append(r)
        print(f"[C2][{device}] {r['config']:24s} p50={r['p50_us']:9.1f} us  {r['gbps']:6.1f} GB/s", flush=True)
    return out


# ---------------------------------------------------------------------------
# C3 — prefix radix-tree byte scan + hash
# ---------------------------------------------------------------------------

C3_CONFIGS = [
    ("prefix_64KB", 64 * 1024),
    ("prefix_256KB", 256 * 1024),
    ("prefix_1MB", 1024 * 1024),
    ("prefix_4MB", 4 * 1024 * 1024),
]


def bench_c3(device: str, dtype: torch.dtype):
    """
    Approximation: byte-level eq comparison + xor-fold hash.
    AMX path can use VNNI 64B compare via _mm512_cmpeq_epu8 + bitscan.
    Without writing intrinsics here, we use torch byte ops to estimate
    memory bandwidth needs; the GPU path is degenerate (host-side op).
    """
    out = []
    is_cuda = device.startswith("cuda")
    for name, sz in C3_CONFIGS:
        torch.manual_seed(0)
        a = torch.randint(0, 256, (sz,), dtype=torch.uint8).to(device=device)
        b = torch.randint(0, 256, (sz,), dtype=torch.uint8).to(device=device)

        def step(a=a, b=b):
            # byte eq + bitwise xor-fold (proxy for compare + rolling hash)
            eq = (a == b).to(torch.int32)
            h = (a ^ b).to(torch.int32)
            return eq.sum() + h.sum()

        r = _measure(step, is_cuda)
        r["config"] = name
        r["bytes"] = 2 * sz  # read a + read b
        r["gbps"] = r["bytes"] / (r["p50_us"] * 1e-6) / 1e9
        out.append(r)
        print(f"[C3][{device}] {r['config']:16s} p50={r['p50_us']:9.1f} us  {r['gbps']:6.1f} GB/s", flush=True)
    return out


# ---------------------------------------------------------------------------
# C4 — KV scale calib: per-layer per-head bf16 abs-max
# ---------------------------------------------------------------------------

# (num_layers, num_heads, head_dim, tokens) — typical Llama-8B layout w/o batching
C4_CONFIGS = [
    # (name, num_heads, head_dim, tokens)
    ("kvscale_h32_d128_t1k", 32, 128, 1024),
    ("kvscale_h32_d128_t8k", 32, 128, 8192),
    ("kvscale_h64_d128_t8k", 64, 128, 8192),
]


def bench_c4(device: str, dtype: torch.dtype):
    out = []
    is_cuda = device.startswith("cuda")
    for name, H, D, T in C4_CONFIGS:
        torch.manual_seed(0)
        # KV cache shape (T, H, D) — abs-max per head_idx
        kv = torch.randn(T, H, D, dtype=torch.float32).to(device=device, dtype=dtype)

        def step(kv=kv):
            return kv.abs().amax(dim=(0, 2))  # → [H]

        r = _measure(step, is_cuda)
        r["config"] = name
        r["bytes"] = T * H * D * 2  # bf16 read
        r["gbps"] = r["bytes"] / (r["p50_us"] * 1e-6) / 1e9
        out.append(r)
        print(f"[C4][{device}] {r['config']:24s} p50={r['p50_us']:9.1f} us  {r['gbps']:6.1f} GB/s", flush=True)
    return out


# ---------------------------------------------------------------------------
# C5 — fused norm + residual add
# ---------------------------------------------------------------------------

C5_CONFIGS = [
    ("tok2k_h4096", 2048, 4096),
    ("tok8k_h4096", 8192, 4096),
    ("tok8k_h8192", 8192, 8192),
]


def bench_c5(device: str, dtype: torch.dtype):
    out = []
    is_cuda = device.startswith("cuda")
    eps = 1e-6
    for name, tok, hidden in C5_CONFIGS:
        torch.manual_seed(0)
        x = torch.randn(tok, hidden, dtype=torch.float32).to(device=device, dtype=dtype)
        res = torch.randn(tok, hidden, dtype=torch.float32).to(device=device, dtype=dtype)
        w = torch.randn(hidden, dtype=torch.float32).to(device=device, dtype=dtype)

        def step(x=x, res=res, w=w):
            s = (x + res).to(torch.float32)
            v = s.pow(2).mean(-1, keepdim=True)
            sn = s * torch.rsqrt(v + eps)
            return (sn.to(dtype)) * w, s.to(dtype)

        r = _measure(step, is_cuda)
        r["config"] = name
        # read x + read res + read w + write y + write s ≈ 5 × tok × hidden × 2
        r["bytes"] = 5 * tok * hidden * 2
        r["gbps"] = r["bytes"] / (r["p50_us"] * 1e-6) / 1e9
        out.append(r)
        print(f"[C5][{device}] {r['config']:24s} p50={r['p50_us']:9.1f} us  {r['gbps']:6.1f} GB/s", flush=True)
    return out


# ---------------------------------------------------------------------------
# CPU stall proxy: run lane fn while a calibrate fn runs on the *same* core
# ---------------------------------------------------------------------------

def stall_proxy(lane_fn, calibrate_us_target=1000):
    """
    Phase-1 style ortho-lane test: while one thread runs the lane op,
    another thread on a different core runs a fixed-cost compute loop.
    Free-fraction = calibrate_rate_during / calibrate_rate_baseline.
    For CPU-bound lanes (AMX matmul/norm), this will be ≪ 1.
    For DSA copy, this was ~1.0.
    Here we report op-level CPU consumption ratio (lane / calibrate) as a proxy.
    """
    import threading

    stop = threading.Event()
    counter = [0]

    def calibrate():
        c = 0
        while not stop.is_set():
            # arithmetic that won't get DCE
            c = (c * 1664525 + 1013904223) & 0xFFFFFFFF
            counter[0] = c
        return c

    # baseline rate
    t0 = time.perf_counter_ns()
    end = t0 + int(50e6)  # 50 ms
    base = 0
    c = 0
    while time.perf_counter_ns() < end:
        c = (c * 1664525 + 1013904223) & 0xFFFFFFFF
        base += 1
    base_rate = base / 0.05

    # during lane
    t = threading.Thread(target=calibrate)
    t.start()
    lane_fn()
    stop.set()
    t.join()

    # not a perfect overlap-measurement; report only relative info
    return {"baseline_rate": base_rate, "calibrate_counter": counter[0]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


SUB_LANES = {
    "c1": ("draft_head_matmul", bench_c1),
    "c2": ("rms_norm", bench_c2),
    "c3": ("prefix_byte_scan", bench_c3),
    "c4": ("kv_scale_calib", bench_c4),
    "c5": ("fused_norm_add", bench_c5),
}


def run_backend(backend: str, lanes: list[str], out_path: Path):
    if backend == "amx":
        device, dtype = "cpu", torch.bfloat16
    elif backend == "avx512":
        device, dtype = "cpu", torch.bfloat16
    elif backend == "gpu":
        device, dtype = "cuda:0", torch.bfloat16
    else:
        raise ValueError(backend)

    # thread cap (AMX → all-cores; AVX-512 same; GPU irrelevant)
    if backend in ("amx", "avx512"):
        nthr = int(os.environ.get("OMP_NUM_THREADS", "56"))
        torch.set_num_threads(nthr)

    print(f"[bench] backend={backend} device={device} dtype={dtype} "
          f"threads={torch.get_num_threads()}", flush=True)

    out = {"backend": backend, "lanes": {}}
    for k in lanes:
        if k not in SUB_LANES:
            continue
        name, fn = SUB_LANES[k]
        print(f"\n=== {k.upper()} {name} ===", flush=True)
        out["lanes"][k] = {"name": name, "results": fn(device, dtype)}

    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[bench] wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["amx", "avx512", "gpu", "all"], default="all")
    ap.add_argument("--lanes", default="c1,c2,c3,c4,c5")
    ap.add_argument("--out-dir", type=Path, default=Path("lhc_phase3"))
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    lanes = args.lanes.split(",")

    if args.backend == "all":
        # gpu first (fastest), then amx (default ISA = AMX on EMR/SPR/GNR),
        # then avx512 via child re-exec.
        for be in ("gpu", "amx"):
            run_backend(be, lanes, args.out_dir / f"amx_sub_lane_{be}.json")
        # avx512 re-exec
        env = os.environ.copy()
        env["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
        env["LHC_BENCH_BACKEND_FORCE"] = "avx512"
        subprocess.run([
            sys.executable, __file__,
            "--backend", "avx512", "--lanes", args.lanes,
            "--out-dir", str(args.out_dir),
        ], env=env, check=True)
    else:
        run_backend(args.backend, lanes, args.out_dir / f"amx_sub_lane_{args.backend}.json")


if __name__ == "__main__":
    main()
