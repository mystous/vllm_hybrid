#!/usr/bin/env python3
"""PLN_004 E1 — AMX expert GEMM microbench (H1: roofline knee).

KTMoEWrapper (kt-kernel 0.7.0.post2) 를 단독 구동해 tokens/expert (n_e) 에 따른
expert 연산 시간 곡선을 측정한다.

라우팅 모드:
  conc  — 모든 토큰이 같은 k개 expert 로 (n_e = T, knee 직접 관측)
  unif  — 균등 랜덤 라우팅 (n_e ≈ T*k/E, serving 근사)

usage: e1_amx_microbench.py <weight_path> <method> <out_json> [E] [k] [h] [m]
"""
import json
import sys
import time

import torch
from kt_kernel.experts import KTMoEWrapper

WPATH, METHOD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
E = int(sys.argv[4]) if len(sys.argv) > 4 else 128
K_ACT = int(sys.argv[5]) if len(sys.argv) > 5 else 8
H = int(sys.argv[6]) if len(sys.argv) > 6 else 2048
M_INTER = int(sys.argv[7]) if len(sys.argv) > 7 else 768
LAYER = 0
DEV = "cuda:0"

wrapper = KTMoEWrapper(
    layer_idx=LAYER,
    num_experts=E,
    num_experts_per_tok=K_ACT,
    hidden_size=H,
    moe_intermediate_size=M_INTER,
    gpu_experts_mask=None,
    cpuinfer_threads=96,
    threadpool_count=2,
    weight_path=WPATH,
    chunked_prefill_size=8192,
    method=METHOD,
    num_gpu_experts=0,
)
wrapper.load_weights(torch.arange(E, dtype=torch.int64))
print("weights loaded", flush=True)

stream = torch.cuda.current_stream(torch.device(DEV)).cuda_stream
results = []

def run(T, mode, iters):
    x = torch.randn(T, H, dtype=torch.float16, device=DEV)
    w = torch.full((T, K_ACT), 1.0 / K_ACT, dtype=torch.float32, device=DEV)
    if mode == "conc":
        ids = torch.arange(K_ACT, dtype=torch.int64, device=DEV).repeat(T, 1)
    else:
        ids = torch.stack([torch.randperm(E, device=DEV)[:K_ACT] for _ in range(T)]).to(torch.int64)
    # warmup
    for _ in range(3):
        wrapper.submit_forward(x, ids, w, stream)
        wrapper.sync_forward(x, stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        wrapper.submit_forward(x, ids, w, stream)
        wrapper.sync_forward(x, stream)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    n_e = T if mode == "conc" else T * K_ACT / E
    rec = dict(T=T, mode=mode, iters=iters, time_ms=dt * 1e3,
               us_per_token=dt / T * 1e6, n_e=n_e)
    print(rec, flush=True)
    results.append(rec)

for mode in ["conc", "unif"]:
    for T in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        iters = 30 if T <= 64 else 12
        run(T, mode, iters)

json.dump(dict(weight=WPATH, method=METHOD, E=E, k=K_ACT, h=H, m=M_INTER,
               results=results), open(OUT, "w"), indent=1)
print("saved", OUT)
