#!/usr/bin/env python3
"""IDE_057 GRC 상한 계측: CPU 계산이 0 인 빈 핸드오프 왕복 시간 = GRC 가 회수할 수 있는 상한.
모든 expert 를 GPU 상주로 표시(gpu_experts_mask 전부 True)하면 CPU 워커는 할 일이 없고 핸드오프 경로만 남는다.
비교: (a) host 콜백 경로 (KT_CALLBACK_FREE 미설정) (b) callback-free (현행) — 각각 층당 µs.
추가로 cold expert 1~32개일 때의 왕복과 대조해 고정비가 실제 층 시간에서 차지하는 비율을 낸다.
usage: grc_probe.py <out.json>"""
import os, sys, time, json, statistics as st
import torch
from kt_kernel.experts import KTMoEWrapper

E, K, H, M = 160, 8, 6144, 2560
DEV = "cuda:0"
out_f = sys.argv[1]
mode = "cf" if os.environ.get("KT_CALLBACK_FREE") else "callback"

def build(n_gpu, mask):
    w = KTMoEWrapper(layer_idx=0, num_experts=E, num_experts_per_tok=K, hidden_size=H, moe_intermediate_size=M,
                     gpu_experts_mask=mask, cpuinfer_threads=int(os.environ.get("KT_THREADS", "96")),
                     threadpool_count=2, weight_path="/models/kt/qwen3-480b-int4",
                     chunked_prefill_size=8192, method="AMXINT4", num_gpu_experts=n_gpu)
    w.load_weights(torch.arange(E, dtype=torch.int64))
    return w

def roundtrip(w, T, n_cold_ids, iters=200):
    x = torch.randn(max(T, 1), H, dtype=torch.bfloat16, device=DEV)
    wt = torch.full((max(T, 1), K), 1.0 / K, dtype=torch.float32, device=DEV)
    ids = torch.tensor([[n_cold_ids[i % len(n_cold_ids)] for i in range(K)] for _ in range(max(T, 1))],
                       dtype=torch.int64, device=DEV)
    st_ = torch.cuda.current_stream(torch.device(DEV)).cuda_stream
    for _ in range(10):
        w.submit_forward(x, ids, wt, st_); w.sync_forward(x, st_)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        w.submit_forward(x, ids, wt, st_); w.sync_forward(x, st_)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

def roundtrip_graph(w, T, ids_list, iters=200):
    """CUDA graph 로 submit+sync 를 캡처해 재생 — 서빙 경로와 동일 (Python 오버헤드 제외)."""
    x = torch.randn(max(T,1), H, dtype=torch.bfloat16, device=DEV)
    wt = torch.full((max(T,1), K), 1.0/K, dtype=torch.float32, device=DEV)
    ids = torch.tensor([[ids_list[i % len(ids_list)] for i in range(K)] for _ in range(max(T,1))], dtype=torch.int64, device=DEV)
    s_ = torch.cuda.Stream(device=DEV)
    st_ = torch.cuda.current_stream(torch.device(DEV)).cuda_stream
    for _ in range(10):
        w.submit_forward(x, ids, wt, st_); w.sync_forward(x, st_)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            cs = torch.cuda.current_stream(torch.device(DEV)).cuda_stream
            w.submit_forward(x, ids, wt, cs); w.sync_forward(x, cs)
    except Exception as e:
        print("graph capture failed:", repr(e)[:200], flush=True); return None
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

res = {"mode": mode}
# (1) 빈 작업: 전 expert GPU 상주 → CPU 워커 할 일 0
mask_all = torch.ones(E, dtype=torch.bool)
w_empty = build(E, mask_all)
res["empty_handoff_us"] = round(roundtrip(w_empty, 1, list(range(K))), 1)
res["empty_handoff_us_T32"] = round(roundtrip(w_empty, 32, list(range(K))), 1)
gr = roundtrip_graph(w_empty, 1, list(range(K)))
res["empty_handoff_graph_us"] = None if gr is None else round(gr, 1)
gr32 = roundtrip_graph(w_empty, 32, list(range(K)))
res["empty_handoff_graph_us_T32"] = None if gr32 is None else round(gr32, 1)
print(f"[{mode}] empty handoff (CUDA graph replay): T=1 {res['empty_handoff_graph_us']} us, T=32 {res['empty_handoff_graph_us_T32']} us", flush=True)
del w_empty
# (2) cold expert 가 있는 경우 (기준: hot-96 → cold physical id 96..159)
mask_hot = torch.zeros(E, dtype=torch.bool); mask_hot[:96] = True
w = build(96, mask_hot)
for n_cold in (1, 4, 8, 16, 32):
    ids = list(range(96, 96 + n_cold))
    us = roundtrip(w, max(1, n_cold * 1 // K + 1), ids, iters=100)
    res[f"cold{n_cold}_us"] = round(us, 1)
    print(f"[{mode}] cold={n_cold:2d}: {us:8.1f} us   (빈 핸드오프 {res['empty_handoff_us']} us = {res['empty_handoff_us']/us*100:.1f}%)", flush=True)
print(f"[{mode}] empty handoff: T=1 {res['empty_handoff_us']} us, T=32 {res['empty_handoff_us_T32']} us", flush=True)
json.dump(res, open(out_f, "w"), indent=1); print("saved", out_f)
