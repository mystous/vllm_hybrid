#!/usr/bin/env python3
"""SUB_239 — FERRY 메커니즘 수준 A/B (실제 ferry.py + 실제 DSA lane + 실제 GPU H2D).

전체 vLLM serve A/B 는 이 컨테이너에서 불가 (vllm._C 가 CUDA 13 런타임 요구, 컨테이너는
CUDA 12.8 / torch cu128). 그러나 swap-in 핫패스
    k_cpu = stager.stage(k_cpu);  k_gpu = k_cpu.to(device, dtype)
는 vllm._C 커널 없이 torch(cu128, B200 동작 확인)만으로 그대로 재현 가능하다. 본 벤치는
NEO swap-in 의 per-layer staging+H2D 를 실제 FerryStager 로 측정한다.

모드:
  direct : k_cpu.to(device)            — non-pinned(pageable) H2D, FERRY off
  ferry  : stager.stage(k_cpu).to(...)  — pinned bounce + (DSA or CPU) 운반 + H2D
DSA 사용 여부는 VLLM_LHC_DSA 환경변수로 프로세스 단위 토글 (lane 가용성).

사용: VLLM_LHC_DSA=1 VLLM_LHC_DSA_DEV=/dev/dsa/wq1.0 python ferry_vllm_bench.py --nblocks 512
"""
import argparse
import os
import time

import torch

from vllm.v1.lhc.ferry import FerryStager


def make_layer(nblocks, kv_heads, block, head_dim, dtype):
    # NEO gather 결과 모사: contiguous, non-pinned CPU 텐서 (HND)
    return torch.randn(nblocks, kv_heads, block, head_dim, dtype=dtype)


def run(mode, layers, tensors, gpu_dtype, dev, stager=None):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for t in tensors:
        if mode == "direct":
            g = t.to(device=dev, dtype=gpu_dtype, non_blocking=True)
        else:
            staged = stager.stage(t)
            g = staged.to(device=dev, dtype=gpu_dtype, non_blocking=True)
        out = g
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)        # Qwen2.5-7B
    ap.add_argument("--nblocks", type=int, default=512)       # swap-in 당 블록 수
    ap.add_argument("--kv-heads", type=int, default=2)        # TP=2 → 4/2
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    src_dtype = torch.float16
    gpu_dtype = torch.bfloat16     # NEO GPU KV dtype (다를 수 있음 → cast)

    # K, V × layers 텐서 풀 (재사용, 매 iter 동일 데이터)
    tensors = []
    for _ in range(args.layers):
        tensors.append(make_layer(args.nblocks, args.kv_heads, args.block, args.head_dim, src_dtype))
        tensors.append(make_layer(args.nblocks, args.kv_heads, args.block, args.head_dim, src_dtype))

    per_layer_mb = args.nblocks * args.kv_heads * args.block * args.head_dim * 2 / (1 << 20)
    total_mb = per_layer_mb * len(tensors)
    dsa_env = os.environ.get("VLLM_LHC_DSA", "0")
    print(f"[cfg] layers={args.layers} (K+V={len(tensors)} tensors) nblocks={args.nblocks} "
          f"per_tensor={per_layer_mb:.2f}MB total={total_mb:.1f}MB/swap-in  VLLM_LHC_DSA={dsa_env}")

    stager = FerryStager()

    # 정확성: ferry-staged 결과가 direct 와 GPU 상에서 bit-exact 동일한지
    t0 = tensors[0]
    g_direct = t0.to(device=dev, dtype=gpu_dtype)
    g_ferry = stager.stage(t0).to(device=dev, dtype=gpu_dtype)
    bit_exact = torch.equal(g_direct, g_ferry)
    print(f"[correctness] ferry vs direct GPU tensor bit-exact: {bit_exact}")

    # warmup
    for _ in range(args.warmup):
        run("direct", args.layers, tensors, gpu_dtype, dev)
        run("ferry", args.layers, tensors, gpu_dtype, dev, stager)

    def bench(mode):
        ts = []
        for _ in range(args.iters):
            dt, _ = run(mode, args.layers, tensors, gpu_dtype, dev, stager)
            ts.append(dt)
        ts.sort()
        return ts[len(ts)//2], sum(ts)/len(ts), ts[0]

    d_med, d_mean, d_min = bench("direct")
    f_med, f_mean, f_min = bench("ferry")

    st = stager.stats
    print(f"[stats] FerryStager dsa_ops={st['dsa_ops']} cpu_ops={st['cpu_ops']}")
    print(f"[direct] median={d_med*1e3:.3f}ms mean={d_mean*1e3:.3f}ms min={d_min*1e3:.3f}ms  "
          f"({total_mb/d_med/1e3:.1f} GB/s)")
    print(f"[ferry ] median={f_med*1e3:.3f}ms mean={f_mean*1e3:.3f}ms min={f_min*1e3:.3f}ms  "
          f"({total_mb/f_med/1e3:.1f} GB/s)")
    delta = (f_med/d_med - 1) * 100
    print(f"[A/B] ferry vs direct median: {delta:+.1f}%  "
          f"({'ferry faster' if delta<0 else 'ferry slower'})")
    print(f"FERRY_BENCH,dsa={dsa_env},nblocks={args.nblocks},total_mb={total_mb:.1f},"
          f"direct_ms={d_med*1e3:.3f},ferry_ms={f_med*1e3:.3f},delta_pct={delta:.1f},"
          f"dsa_ops={st['dsa_ops']},cpu_ops={st['cpu_ops']},bit_exact={bit_exact}")


if __name__ == "__main__":
    main()
