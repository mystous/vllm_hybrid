#!/usr/bin/env bash
# random 토큰 워크로드의 expert 라우팅 트레이스 수집 (recorder per_pass) → routing_stats_random.json
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
until grep -q "COLDDEFER_KILL_DONE" /tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad/colddefer.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$HOME/projects/vllm_hybrid/eval/results/$(date +%Y%m%d_%H%M%S)_pln008_routing_trace_random; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; rm -f /tmp/expert_distribution_recorder_*.pt; true'; sleep 5
docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 4 --expert-distribution-recorder-mode per_pass --disable-cuda-graph > /tmp/sgl_trace.log 2>&1"
i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo HEALTH | tee $B/RUN.log; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo BOOT_FAIL | tee $B/RUN.log; exit 1; }; sleep 10; i=$((i+10)); done
curl -s -X POST http://127.0.0.1:30000/start_expert_distribution_record >/dev/null; echo "record start" | tee -a $B/RUN.log
rf=/tmp/trace_rand.log; docker exec -d vllm-h100 bash -c "timeout 400 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name random --random-input-len 1000 --random-output-len 128 --random-prefix-len 0 --num-prompts 64 --max-concurrency 32 --request-rate inf --seed 42 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>460)) && break; done
curl -s -X POST http://127.0.0.1:30000/stop_expert_distribution_record >/dev/null; curl -s -X POST http://127.0.0.1:30000/dump_expert_distribution_record >/dev/null; sleep 5
docker exec sgl-kt bash -c 'ls -la /tmp/expert_distribution_recorder_*.pt | head -3' | tee -a $B/RUN.log
docker exec -i sgl-kt python3 - <<'PY' | tee -a $B/RUN.log
import torch, glob, json, statistics
fs=sorted(glob.glob("/tmp/expert_distribution_recorder_*_0.pt")); print("files", len(fs))
d=torch.load(fs[0], map_location="cpu", weights_only=False); tot=None
for r in d["records"]:
    c=r["global_physical_count"]; tot=c.clone().double() if tot is None else tot+c.double()
L,E=tot.shape; tokens=float(tot[0].sum()/8); out={"tokens":tokens,"per_H":{}}
hot=json.load(open("/tmp/hotmap.json"))["physical_to_logical_map"]
for H in (64,80,96,112,128):
    cov=[]; Dh={}; Dc={}
    for B in (8,16,32,64,128):
        dh=[]; dc=[]
        for l in range(L):
            # physical idx 순서가 hotmap 빈도순 → 앞 H 개가 hot (recorder 는 physical 카운트)
            p=(tot[l]/tokens).clamp(max=1.0); ph=p[:H]; pc=p[H:]
            dh.append(float((1-(1-ph)**B).sum())); dc.append(float((1-(1-pc)**B).sum()))
        Dh[B]=round(statistics.mean(dh),2); Dc[B]=round(statistics.mean(dc),2)
    for l in range(L): cov.append(float(tot[l][:H].sum()/tot[l].sum()))
    out["per_H"][H]={"coverage_mean":round(statistics.mean(cov),4),"cold_pairs_per_token":round(8*(1-statistics.mean(cov)),3),"E_distinct_hot":Dh,"E_distinct_cold":Dc}
    print(H, "coverage %.3f cold_pairs/tok %.2f E_Dc(B32) %.1f" % (out["per_H"][H]["coverage_mean"], out["per_H"][H]["cold_pairs_per_token"], Dc[32]))
json.dump(out, open("/tmp/routing_stats_random.json","w"), indent=1); print("saved")
PY
docker cp sgl-kt:/tmp/routing_stats_random.json $B/ 2>/dev/null
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "RANDOM_TRACE_DONE $B"
