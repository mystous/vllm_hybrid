"""
LHC Phase 2 — AMX logits head matmul microbench.

Measures the LM-head GEMM that produces logits from the final hidden state:
    logits[bs, vocab] = hidden[bs, hidden_dim] @ embed_T[hidden_dim, vocab]

This is the AMX sweet spot (BF16 GEMM with K = 4096-8192, N = 128k-256k).
Backends:
  (a) AMX bf16   : torch CPU bf16, oneDNN default ISA (AMX active on EMR)
  (b) AVX-512 BF16: child re-exec with ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16
  (c) GPU bf16   : torch CUDA bf16 on B200, single GPU

Phase 1 sampler bench produced lat ratios ~3-13x because softmax/topk are
memory-bound; here we measure the BF16-GEMM half which is AMX's actual
benefit region.

Gate (Phase 2 plan):
  AMX/GPU latency ratio <= 1.5x  AND  throughput >= 50k logits/s @ vocab 152k.
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

# (vocab_name, vocab_size). Hidden dim from model card.
CONFIGS = [
    ("llama-128k", 128_256, 4096),    # Llama-3.1-8B: hidden 4096
    ("qwen-152k", 152_064, 3584),     # Qwen2.5-7B: hidden 3584
    ("deepseek-256k", 257_152, 4096), # nominal
]
BATCHES = [1, 8, 32, 64]
WARMUP = 5
ITERS = 30


def bench_matmul(
    bs: int, hidden_dim: int, vocab: int, device: str, dtype: torch.dtype
):
    """Returns (median, p10, p90) latency in microseconds."""
    torch.manual_seed(0)
    h = torch.randn(bs, hidden_dim, dtype=torch.float32).to(device=device, dtype=dtype)
    # embed_T is the LM-head weight transposed: [hidden, vocab]
    w = torch.randn(hidden_dim, vocab, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    is_cuda = device.startswith("cuda")

    def step():
        return torch.matmul(h, w)

    for _ in range(WARMUP):
        out = step()
        if is_cuda:
            torch.cuda.synchronize()

    samples = []
    for _ in range(ITERS):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        out = step()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)

    samples.sort()
    p50 = samples[len(samples) // 2]
    p10 = samples[int(len(samples) * 0.10)]
    p90 = samples[int(len(samples) * 0.90)]
    p99 = samples[min(len(samples) - 1, int(len(samples) * 0.99))]
    return p50, p10, p90, p99


def run_backend(backend: str, out_path: Path):
    results = []
    if backend == "amx":
        # Try to confirm AMX is being picked. ipex.verbose if available,
        # otherwise just rely on oneDNN default ISA = avx10_1_512_amx on EMR.
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            print(f"[amx] ipex={ipex.__version__}", flush=True)
        except Exception as e:
            print(f"[amx] ipex unavailable ({e!r}); using oneDNN default (AMX on EMR)",
                  flush=True)

    for vocab_name, vocab, hidden in CONFIGS:
        for bs in BATCHES:
            if backend in ("amx", "avx512"):
                device, dtype = "cpu", torch.bfloat16
            elif backend == "gpu":
                device, dtype = "cuda:0", torch.bfloat16
            else:
                raise ValueError(backend)

            p50, p10, p90, p99 = bench_matmul(bs, hidden, vocab, device, dtype)
            # throughput in logits/s = bs * vocab / lat
            tps_logits = (bs * vocab) / p50 * 1e6
            # also report row-level throughput (bs / lat) for batched generation
            tps_rows = (bs / p50) * 1e6
            row = {
                "backend": backend,
                "vocab_name": vocab_name,
                "vocab": vocab,
                "hidden": hidden,
                "batch": bs,
                "lat_us_p50": round(p50, 2),
                "lat_us_p10": round(p10, 2),
                "lat_us_p90": round(p90, 2),
                "lat_us_p99": round(p99, 2),
                "throughput_logits_per_s": round(tps_logits, 1),
                "throughput_rows_per_s": round(tps_rows, 1),
            }
            results.append(row)
            print(json.dumps(row), flush=True)

    out_path.write_text(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["amx", "avx512", "gpu", "all"], default="all")
    ap.add_argument(
        "--out-dir", default="/workspace/host_vllm_hybrid/lhc_phase2"
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "all":
        print("=== backend: amx (oneDNN default, AMX on EMR) ===", flush=True)
        os.environ.pop("ONEDNN_MAX_CPU_ISA", None)
        run_backend("amx", out_dir / "amx_logitshead_amx.json")

        print(
            "=== backend: avx512 (oneDNN AVX512_CORE_BF16, AMX off) ===", flush=True
        )
        env = os.environ.copy()
        env["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
        subprocess.check_call(
            [
                sys.executable,
                __file__,
                "--backend",
                "avx512",
                "--out-dir",
                str(out_dir),
            ],
            env=env,
        )

        print("=== backend: gpu (CUDA bf16 on B200) ===", flush=True)
        run_backend("gpu", out_dir / "amx_logitshead_gpu.json")
    else:
        suffix = backend = args.backend
        run_backend(args.backend, out_dir / f"amx_logitshead_{suffix}.json")


if __name__ == "__main__":
    main()
