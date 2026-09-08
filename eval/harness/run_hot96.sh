#!/usr/bin/env bash
# (1) DDR 대역폭 상한 측정 (96스레드 STREAM류)  (2) hot-96 부팅 + C=32 벤치
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
HOT="${HOT:-96}"; MF="${MF:-0.85}"
LOG=/tmp/sgl_hot${HOT}${TAG:-}.log
REPO=$HOME/projects/vllm_hybrid
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide030_hot${HOT}${TAG:-}_c32
mkdir -p "$BASE"

echo "== DDR BW $(date +%H:%M:%S) ==" | tee "$BASE/RUN.log"
docker exec -i sgl-kt python3 - <<'EOF' 2>&1 | tee -a "$BASE/RUN.log"
import torch, time, os
torch.set_num_threads(96)
n = 2_000_000_000 // 2  # 2GB bf16
a = torch.ones(n, dtype=torch.bfloat16)
b = torch.empty_like(a)
for it in range(3):
    t0=time.perf_counter(); b.copy_(a); dt=time.perf_counter()-t0
    print(f"copy 2GB (r+w 4GB): {4/dt:.0f} GB/s  (threads={torch.get_num_threads()})")
for it in range(3):
    t0=time.perf_counter(); s=a.sum(); dt=time.perf_counter()-t0
    print(f"read-only sum 2GB: {2/dt:.0f} GB/s")
EOF

docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
echo "== BOOT hot-$HOT mf=$MF $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"
docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static $MF --max-total-tokens ${MTT:-32768} --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $HOT --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic ${EXTRA_ARGS:-} > $LOG 2>&1"
i=0
while ((i<1800)); do
  curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "HEALTH_OK at ${i}s" | tee -a "$BASE/RUN.log"; break; }
  docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || { echo "BOOT_FAIL at ${i}s" | tee -a "$BASE/RUN.log"; docker exec sgl-kt grep -m2 -B2 -A12 "OutOfMemory\|CUDA out of memory\|Traceback" $LOG | head -40 >> "$BASE/RUN.log"; exit 1; }
  sleep 10; i=$((i+10))
done
((i>=1800)) && { echo BOOT_TIMEOUT | tee -a "$BASE/RUN.log"; exit 1; }
docker exec sgl-kt grep -m1 "Capturing batches" $LOG | grep -o "avail_mem=[0-9.]* GB" | head -1 | tee -a "$BASE/RUN.log"
curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"q480","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | python3 -c "import json,sys; print('Q1:', json.load(sys.stdin)['choices'][0]['text'][:60])" | tee -a "$BASE/RUN.log"
echo "== BENCH C=32 $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"
docker exec vllm-h100 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions \
  --model q480 --tokenizer "$TOK" --dataset-name sonnet --dataset-path /tmp/sonnet.txt \
  --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 \
  --num-prompts 128 --max-concurrency 32 --request-rate inf --seed 42 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > "$BASE/bench_c32.log" 2>&1
grep -E "Output token throughput|Mean TTFT|Mean TPOT|Successful" "$BASE/bench_c32.log" | tee -a "$BASE/RUN.log"
echo "== DONE $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"; echo "$BASE"
