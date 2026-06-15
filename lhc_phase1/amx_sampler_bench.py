"""
LHC Phase 1 — AMX sampler microbench.

Measures softmax + top-k=5 latency / throughput on:
  (a) AMX bf16  : torch CPU bf16, oneDNN ISA = avx10_1_512_amx (default on SPR/EMR)
  (b) AVX-512 BF16: torch CPU bf16, oneDNN ISA capped at AVX512_CORE_BF16 (AMX off)
  (c) GPU bf16   : torch CUDA bf16 on B200

Workload mirrors vLLM sampler: logits [batch, vocab] -> softmax -> top-k=5.

The (b) variant uses a child-process re-exec with ONEDNN_MAX_CPU_ISA set, since
oneDNN reads the env once at first primitive creation.
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

VOCABS = {"llama-128k": 128_256, "qwen-152k": 152_064, "deepseek-256k": 257_152}
BATCHES = [1, 8, 32, 64]
WARMUP = 20
ITERS = 200
TOPK = 5


def _bench_once(logits: torch.Tensor, device: str, dtype: torch.dtype) -> float:
    """Returns median latency in microseconds for softmax + top-k."""
    x = logits.to(device=device, dtype=dtype)
    is_cuda = device.startswith("cuda")

    def step():
        p = torch.softmax(x, dim=-1)
        return torch.topk(p, k=TOPK, dim=-1)

    # warmup
    for _ in range(WARMUP):
        v, i = step()
        if is_cuda:
            torch.cuda.synchronize()

    samples = []
    for _ in range(ITERS):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        v, i = step()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)  # μs

    samples.sort()
    med = samples[len(samples) // 2]
    p10 = samples[int(len(samples) * 0.10)]
    p90 = samples[int(len(samples) * 0.90)]
    return med, p10, p90


def run_backend(backend: str, out_path: Path):
    results = []
    for vocab_name, vocab in VOCABS.items():
        for bs in BATCHES:
            torch.manual_seed(0)
            logits = torch.randn(bs, vocab, dtype=torch.float32)

            if backend in ("amx", "avx512"):
                device, dtype = "cpu", torch.bfloat16
            elif backend == "gpu":
                device, dtype = "cuda:0", torch.bfloat16
            else:
                raise ValueError(backend)

            med, p10, p90 = _bench_once(logits, device, dtype)
            tps = (bs / med) * 1e6  # samples per second
            row = {
                "backend": backend,
                "vocab_name": vocab_name,
                "vocab": vocab,
                "batch": bs,
                "lat_us_p50": round(med, 2),
                "lat_us_p10": round(p10, 2),
                "lat_us_p90": round(p90, 2),
                "throughput_sps": round(tps, 1),
            }
            results.append(row)
            print(json.dumps(row), flush=True)

    out_path.write_text(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["amx", "avx512", "gpu", "all"], default="all")
    ap.add_argument("--out-dir", default="/workspace/host_vllm_hybrid/lhc_phase1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "all":
        # Run AMX in-process
        print("=== backend: amx (oneDNN AMX, default) ===", flush=True)
        os.environ.pop("ONEDNN_MAX_CPU_ISA", None)
        run_backend("amx", out_dir / "amx_sampler_amx.json")

        # Run AVX-512 in a child process to bypass oneDNN ISA caching
        print("=== backend: avx512 (oneDNN AVX512_CORE_BF16, AMX off) ===", flush=True)
        env = os.environ.copy()
        env["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
        subprocess.check_call(
            [sys.executable, __file__, "--backend", "avx512", "--out-dir", str(out_dir)],
            env=env,
        )

        # Run GPU
        print("=== backend: gpu (CUDA bf16 on B200) ===", flush=True)
        run_backend("gpu", out_dir / "amx_sampler_gpu.json")
    else:
        suffix = {"amx": "amx", "avx512": "avx512", "gpu": "gpu"}[args.backend]
        run_backend(args.backend, out_dir / f"amx_sampler_{suffix}.json")


if __name__ == "__main__":
    main()
