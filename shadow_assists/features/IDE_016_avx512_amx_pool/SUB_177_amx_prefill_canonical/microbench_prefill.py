"""SUB_177 — AMX prefill kernel microbench (medium-context 512/1024/2048).

prefill 영역에서 의미 있는 shape (Qwen 32B MLP/QKV)에 대해 본 AMX kernel 의
single-thread throughput 측정. 동일 shape 에 PyTorch CPU multi-thread 비교.

목적:
  - 본 naive AMX kernel 의 prefill 적용 가능여부 (TTFT 보조)
  - SUB_175 의 0.046-0.052 TFLOPS 한계 와의 정합성 확인
  - paper §4 TSK_027 의 TTFT −15% 1K context target 의 feasibility 사전 평가

shapes (medium-context prefill, batch=context_length):
  - Qwen 32B hidden=5120, intermediate=27648
  - prefill 의 매 layer 별 [B, K] · [K, N] → [B, N]
  - B = 512 / 1024 / 2048 (context length)
"""
from __future__ import annotations

import argparse, json, os, sys, time

# AMX 와 OMP fork 격리
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


def bench(fn, warmup=2, iters=3):
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


# Qwen 32B MLP (representative prefill matmul)
SHAPES = {
    "qwen32b_gate_up": (5120, 27648),
    "qwen32b_down":    (27648, 5120),
    "qwen32b_qkv":     (5120, 7680),    # QKV proj (Q=5120 + K=1280 + V=1280, GQA 8:1)
    "qwen32b_o_proj":  (5120, 5120),
}

# context lengths (prefill batch sizes for one request)
B_LENGTHS = [512, 1024, 2048]


def run_amx():
    avx512_amx_pool.matmul.request_amx_permission()
    print(f"[AMX] avail={avx512_amx_pool.amx_is_available()} threads={torch.get_num_threads()}")
    rows = []
    for shape_name, (K, N) in SHAPES.items():
        for B in B_LENGTHS:
            A = torch.randn(B, K, dtype=torch.bfloat16)
            Bmat = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
            B_packed = avx512_amx_pool.matmul.amx_repack_b(Bmat)

            def amx_run():
                return avx512_amx_pool.matmul.amx_matmul(A, B_packed)

            try:
                best, avg = bench(amx_run, warmup=1, iters=3)
                tf = tflops(B, K, N, best)
            except Exception as e:
                best, avg, tf = -1, -1, -1
                print(f"  ERR {shape_name} B={B}: {e}")
            rows.append({
                "kernel": "amx",
                "shape": shape_name, "B": B, "K": K, "N": N,
                "best_ms": best * 1000, "avg_ms": avg * 1000,
                "tflops": tf,
            })
            print(f"  AMX {shape_name} B={B:>5} {K}x{N}: best={best*1000:.2f}ms TFLOPS={tf:.4f}")
    return rows


def run_pt(threads):
    torch.set_num_threads(threads)
    print(f"[PT] threads={torch.get_num_threads()}")
    rows = []
    for shape_name, (K, N) in SHAPES.items():
        for B in B_LENGTHS:
            A = torch.randn(B, K, dtype=torch.bfloat16)
            Bmat = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

            def pt_run():
                return torch.matmul(A, Bmat)

            best, avg = bench(pt_run, warmup=2, iters=3)
            tf = tflops(B, K, N, best)
            rows.append({
                "kernel": f"pt_t{threads}",
                "shape": shape_name, "B": B, "K": K, "N": N,
                "best_ms": best * 1000, "avg_ms": avg * 1000,
                "tflops": tf,
            })
            print(f"  PT-{threads}thr {shape_name} B={B:>5} {K}x{N}: best={best*1000:.2f}ms TFLOPS={tf:.4f}")
    return rows


def estimate_layer_latency(rows, layer_count=64):
    """Qwen 32B = 64 layers. Per-layer = (QKV + O) for attention + (gate_up + down) for MLP.
    Per-layer total ≈ 1 QKV + 1 O + 1 gate_up + 1 down.
    """
    print("\n=== per-layer estimate (Qwen 32B, 64 layers) ===")
    # Group by kernel/B
    by_kb = {}
    for r in rows:
        key = (r["kernel"], r["B"])
        by_kb.setdefault(key, {})[r["shape"]] = r["best_ms"]

    summary = []
    for (kernel, B), shape_map in sorted(by_kb.items()):
        ms_per_layer = (
            shape_map.get("qwen32b_qkv", 0)
            + shape_map.get("qwen32b_o_proj", 0)
            + shape_map.get("qwen32b_gate_up", 0)
            + shape_map.get("qwen32b_down", 0)
        )
        total_ms = ms_per_layer * layer_count
        print(f"  {kernel:>10} B={B:>5}: per_layer={ms_per_layer:.2f}ms total_64L={total_ms/1000:.2f}s")
        summary.append({"kernel": kernel, "B": B,
                        "per_layer_ms": ms_per_layer, "total_ms": total_ms,
                        "total_sec": total_ms / 1000.0})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["amx", "pt1", "pt4", "pt32", "all"], default="all")
    args = parser.parse_args()

    all_rows = []
    if args.mode in ("amx", "all"):
        all_rows.extend(run_amx())
    if args.mode in ("pt1", "all"):
        all_rows.extend(run_pt(1))
    if args.mode in ("pt4", "all"):
        all_rows.extend(run_pt(4))
    if args.mode in ("pt32",):
        all_rows.extend(run_pt(32))

    summary = estimate_layer_latency(all_rows)

    with open(args.out, "w") as f:
        json.dump({"rows": all_rows, "summary": summary}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
