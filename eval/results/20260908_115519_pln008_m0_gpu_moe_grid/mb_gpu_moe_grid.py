#!/usr/bin/env python3
"""M0 파라미터: GPU hot expert 커널 g(H, B, D_h) 단독 격자 벤치 (sglang fused_experts, FP8 w8a8 block[128,128]).

Qwen3-480B TP=4 shard: hidden 6144, intermediate/TP = 640 → w1 [E, 1280, 6144], w2 [E, 6144, 640] (fp8 e4m3).
축: H(=E) ∈ {64,80,96,112}, B(토큰) ∈ {8,16,32,64}, 라우팅: 실제 hot 분포 모사 — 각 토큰 k=8 중
hot 선택 비율 97~98.5% 를 H 개 expert 위 zipf 로 뽑음 (distinct hot 수 D_h 를 함께 기록).
usage: mb_gpu_moe_grid.py <out_json>
"""
import json, sys, time, math, glob
import torch
# sglang 설정 네임스페이스 publish (fused_experts 가 exec 네임스페이스를 읽음)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
_mp = sorted(glob.glob("/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*"))[0]
_sa = ServerArgs(model_path=_mp, tp_size=1, attention_backend="triton", disable_cuda_graph=True)
set_global_server_args_for_scheduler(_sa)
from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29517", local_rank=0, backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

OUT = sys.argv[1]
dev = "cuda:0"; HID = 6144; INTER = 640; K = 8
BS = 128
torch.manual_seed(0)

def make_weights(E):
    w1 = (torch.randn(E, 2*INTER, HID, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w2 = (torch.randn(E, HID, INTER, device=dev) * 0.05).to(torch.float8_e4m3fn)
    s1 = torch.rand(E, math.ceil(2*INTER/BS), math.ceil(HID/BS), device=dev, dtype=torch.float32) * 0.01
    s2 = torch.rand(E, math.ceil(HID/BS), math.ceil(INTER/BS), device=dev, dtype=torch.float32) * 0.01
    return w1, w2, s1, s2

def routing(E, B, zipf_a=1.1):
    # zipf 빈도로 H 개 expert 중 k 개 비복원 추출 (hot 집중 모사)
    p = torch.tensor([1.0/((i+1)**zipf_a) for i in range(E)]); p = p/p.sum()
    ids = torch.stack([torch.multinomial(p, K, replacement=False) for _ in range(B)]).to(dev)
    w = torch.softmax(torch.randn(B, K, device=dev), dim=-1)
    return ids, w, int(ids.unique().numel())

res = []
for E in (64, 80, 96, 112):
    w1, w2, s1, s2 = make_weights(E)
    cfg = MoeRunnerConfig(num_experts=E, num_local_experts=E, hidden_size=HID,
                          intermediate_size_per_partition=INTER, top_k=K, params_dtype=torch.bfloat16,
                          activation="silu", inplace=False)
    for B in (8, 16, 32, 64):
        x = torch.randn(B, HID, device=dev, dtype=torch.bfloat16)
        ids, w, dh = routing(E, B)
        topk = StandardTopKOutput(topk_weights=w, topk_ids=ids.to(torch.int32), router_logits=torch.zeros(B, E, device=dev))
        def run():
            return fused_experts(x, w1, w2, topk, cfg, use_fp8_w8a8=True, w1_scale=s1, w2_scale=s2, block_shape=[BS, BS])
        try:
            # eager 워밍업 → CUDA graph 캡처 → 재생 시간 (서빙과 같은 graph 재생 조건; launch 간격 제외)
            st = torch.cuda.Stream()
            with torch.cuda.stream(st):
                for _ in range(3): run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st):
                out = run()
            torch.cuda.synchronize()
            for _ in range(5): g.replay()
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(50): g.replay()
            ev1.record(); torch.cuda.synchronize()
            us = ev0.elapsed_time(ev1) / 50 * 1000
            # eager 비교치
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20): run()
            e1.record(); torch.cuda.synchronize()
            us_eager = e0.elapsed_time(e1) / 20 * 1000
            rec = dict(H=E, B=B, distinct_hot=dh, us_per_layer_graph=round(us, 1), us_per_layer_eager=round(us_eager, 1), bytes_touched_MB=round(dh*(2*INTER*HID+HID*INTER)/1e6, 1))
        except Exception as e:
            rec = dict(H=E, B=B, error=str(e)[:160])
        print(rec, flush=True); res.append(rec)
    del w1, w2, s1, s2; torch.cuda.empty_cache()
json.dump(dict(hidden=HID, inter_tp=INTER, k=K, block=BS, results=res), open(OUT, "w"), indent=1)
print("saved", OUT)
