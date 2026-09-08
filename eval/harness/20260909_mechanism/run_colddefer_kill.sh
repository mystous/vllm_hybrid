#!/usr/bin/env bash
# 배치-인지 deferral kill test: hot-96, N ∈ {8, 6} (cold 전량/대부분 deferral) → C=32 처리량 + greedy + GSM40
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$HOME/projects/vllm_hybrid/eval/results/$(date +%Y%m%d_%H%M%S)_ide032_colddefer_kill; mkdir -p $B
for N in 8 6; do
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token $N > /tmp/sgl_cd$N.log 2>&1"
  i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  [ $ok = 1 ] || { echo "N=$N BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m1 -A5 -E "Error|assert" /tmp/sgl_cd$N.log | tail -6 >> $B/RUN.log; continue; }
  echo "== N=$N boot ok ${i}s ==" | tee -a $B/RUN.log
  for P in "The capital of France is" "def fibonacci(n):\n    "; do curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"q480\",\"prompt\":\"$P\",\"max_tokens\":24,\"temperature\":0}" | python3 -c "import json,sys; print('  greedy:', repr(json.load(sys.stdin)['choices'][0]['text'][:70]))" | tee -a $B/RUN.log; done
  rf=/tmp/cd${N}_c32.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 128 --max-concurrency 32 --request-rate inf --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/N${N}_c32.log 2>/dev/null
  echo "N=$N C32: $(grep -h 'Output token throughput' $B/N${N}_c32.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/N${N}_c32.log|awk '{print $NF}') ms" | tee -a $B/RUN.log
  python3 $SP/gsm_eval.py $B/N${N}_gsm40.json 40 2>&1 | tail -1 | sed "s/^/N=$N GSM40: /" | tee -a $B/RUN.log
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "COLDDEFER_KILL_DONE $B"
