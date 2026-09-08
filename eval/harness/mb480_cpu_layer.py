#!/usr/bin/env python3
"""IDE_030 후속 — 480B AMXINT4 층 1회 CPU 호출의 순수 비용 분해.

serving 근사: T 토큰, k=8, 그중 cold(=CPU) expert 가 n_cold 개로 집중되도록 라우팅.
submit_forward→sync_forward 왕복(=graph 안 host 노드/복사/큐/커널 전부 포함) 시간을 잰다.
usage: mb480_cpu_layer.py <out_json>
env: KT_THREADS(96) KT_POOLS(2)
"""
import json, os, sys, time
import torch
from kt_kernel.experts import KTMoEWrapper

OUT = sys.argv[1]
E, K, H, M = 160, 8, 6144, 2560
WPATH = "/models/kt/qwen3-480b-int4"
DEV = "cuda:0"

wrapper = KTMoEWrapper(
    layer_idx=0, num_experts=E, num_experts_per_tok=K, hidden_size=H,
    moe_intermediate_size=M, gpu_experts_mask=None,
    cpuinfer_threads=int(os.environ.get("KT_THREADS", "96")),
    threadpool_count=int(os.environ.get("KT_POOLS", "2")),
    weight_path=WPATH, chunked_prefill_size=8192, method="AMXINT4", num_gpu_experts=0,
)
wrapper.load_weights(torch.arange(E, dtype=torch.int64))
print("weights loaded", flush=True)
stream = torch.cuda.current_stream(torch.device(DEV)).cuda_stream
results = []

def run(T, n_cold, iters=40):
    """T 토큰 각각 k=8 expert 선택, 전체가 n_cold 개의 expert 안에서만 고르게 분포."""
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV)
    w = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(0)
    if n_cold <= K:
        ids = torch.arange(n_cold, dtype=torch.int64).repeat(T, (K + n_cold - 1) // n_cold)[:, :K]
    else:
        ids = torch.stack([torch.randperm(n_cold, generator=g)[:K] for _ in range(T)])
    ids = ids.to(DEV)
    for _ in range(5):
        wrapper.submit_forward(x, ids, w, stream); wrapper.sync_forward(x, stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        wrapper.submit_forward(x, ids, w, stream); wrapper.sync_forward(x, stream)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    rec = dict(T=T, n_cold=n_cold, iters=iters, us=round(dt * 1e6, 1),
               us_per_expert=round(dt * 1e6 / n_cold, 1))
    print(rec, flush=True); results.append(rec)

for T in (32, 16, 1):
    for n in (1, 2, 4, 7, 12, 24, 48):
        run(T, n)
json.dump(dict(E=E, k=K, h=H, m=M, results=results), open(OUT, "w"), indent=1)
print("saved", OUT)
