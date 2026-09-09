#!/usr/bin/env python3
"""IDE_046 — prefill 규모 rows 스윕 (AMX mat_mul 경로). n_cold 8·64, rows/expert 32~256.
usage: mb480_rows3.py <out_json>   env: KT_THREADS(96) KT_POOLS(2) KT_AMX_MIN_QLEN=0 (모든 qlen 을 AMX 경로로)"""
import json, os, sys, time
import torch
from kt_kernel.experts import KTMoEWrapper

OUT = sys.argv[1]
E, K, H, M = 160, 8, 6144, 2560
wrapper = KTMoEWrapper(layer_idx=0, num_experts=E, num_experts_per_tok=K, hidden_size=H,
    moe_intermediate_size=M, gpu_experts_mask=None,
    cpuinfer_threads=int(os.environ.get("KT_THREADS", "96")), threadpool_count=int(os.environ.get("KT_POOLS", "2")),
    weight_path="/models/kt/qwen3-480b-int4", chunked_prefill_size=8192, method="AMXINT4", num_gpu_experts=0)
wrapper.load_weights(torch.arange(E, dtype=torch.int64)); print("weights loaded", flush=True)
DEV = "cuda:0"; stream = torch.cuda.current_stream(torch.device(DEV)).cuda_stream
results = []
def run(T, n_cold, iters):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV)
    w = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(0)
    # T 토큰 × k=8 을 n_cold 개 expert 에 정확히 고르게: rows/expert = 8T/n_cold
    flat = torch.arange(T * K) % n_cold
    flat = flat[torch.randperm(T * K, generator=g)]
    ids = flat.view(T, K)
    # 한 토큰 안의 중복 expert 는 허용 (kt 는 행 단위로 취급)
    ids = ids.to(DEV)
    for _ in range(3):
        wrapper.submit_forward(x, ids, w, stream); wrapper.sync_forward(x, stream)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        wrapper.submit_forward(x, ids, w, stream); wrapper.sync_forward(x, stream)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / iters
    rows = T * K / n_cold
    rec = dict(T=T, n_cold=n_cold, rows=rows, us=round(dt * 1e6, 1), us_per_expert=round(dt * 1e6 / n_cold, 1),
               us_per_row=round(dt * 1e6 / n_cold / rows, 2))
    print("  n_cold=%3d T=%5d rows/expert=%6.1f us=%9.1f us/expert=%8.1f us/row=%5.2f" % (
        n_cold, T, rows, rec["us"], rec["us_per_expert"], rec["us_per_row"]), flush=True); results.append(rec)
for n in (8, 64):
    for rows in (16, 32, 64, 128, 256):
        T = rows * n // K
        run(T, n, iters=20 if T <= 512 else 8)
json.dump(dict(E=E, k=K, h=H, m=M, results=results), open(OUT, "w"), indent=1); print("saved", OUT)
