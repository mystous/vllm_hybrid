#!/usr/bin/env python3
"""IDE_046 수치 검증: 같은 입력 (seed 고정) 에 대한 층 출력을 파일로 저장. base / bcache 두 프로세스 결과를 비교.
usage: mb480_numcheck.py <out_pt>   (env 로 KT_AMX_BCACHE 등 제어; KT_AMX_MIN_QLEN=0 으로 AMX 경로 강제)"""
import os, sys, torch
from kt_kernel.experts import KTMoEWrapper
E, K, H, M = 160, 8, 6144, 2560
wrapper = KTMoEWrapper(layer_idx=0, num_experts=E, num_experts_per_tok=K, hidden_size=H, moe_intermediate_size=M,
    gpu_experts_mask=None, cpuinfer_threads=96, threadpool_count=2, weight_path="/models/kt/qwen3-480b-int4",
    chunked_prefill_size=8192, method="AMXINT4", num_gpu_experts=0)
wrapper.load_weights(torch.arange(E, dtype=torch.int64))
DEV = "cuda:0"; stream = torch.cuda.current_stream(torch.device(DEV)).cuda_stream
outs = {}
for T, n_cold in ((8, 8), (40, 8), (200, 8), (1024, 64)):
    g = torch.Generator(device="cpu").manual_seed(1234 + T)
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    w = torch.softmax(torch.randn(T, K, generator=g), -1).to(DEV)
    flat = (torch.arange(T * K) % n_cold)[torch.randperm(T * K, generator=g)]
    ids = flat.view(T, K).to(DEV)
    wrapper.submit_forward(x, ids, w, stream); y = wrapper.sync_forward(x, stream)
    torch.cuda.synchronize(); outs[(T, n_cold)] = y.detach().float().cpu().clone()
    print("T=%d n_cold=%d out mean|y|=%.4f" % (T, n_cold, outs[(T, n_cold)].abs().mean()), flush=True)
torch.save(outs, sys.argv[1]); print("saved", sys.argv[1])
