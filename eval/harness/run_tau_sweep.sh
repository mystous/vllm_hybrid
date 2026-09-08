#!/usr/bin/env bash
# IDE_032 τ 스윕: hot-96, KT_COLD_DEFER=1, τ ∈ {1.0, 0.3, 0.15} (+ N 은 8 로 두어 protected 경로 우회) → C32 + GSM40
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
until grep -q "N8_PROFILE_DONE\|BOOT_FAIL" $SP/n8prof.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$HOME/projects/vllm_hybrid/eval/results/$(date +%Y%m%d_%H%M%S)_ide032_tau_sweep; mkdir -p $B
for TAU in 1.0 0.3 0.15; do
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "KT_COLD_DEFER=1 KT_COLD_TAU=$TAU CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_tau$TAU.log 2>&1"
  i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  [ $ok = 1 ] || { echo "tau=$TAU BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m1 -A5 -E "Error|assert" /tmp/sgl_tau$TAU.log | tail -6 >> $B/RUN.log; continue; }
  echo "== tau=$TAU boot ${i}s ==" | tee -a $B/RUN.log
  curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d '{"model":"q480","prompt":"The capital of France is","max_tokens":16,"temperature":0}' | python3 -c "import json,sys; print('  greedy:', repr(json.load(sys.stdin)['choices'][0]['text'][:60]))" | tee -a $B/RUN.log
  rf=/tmp/tau${TAU}_c32.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 128 --max-concurrency 32 --request-rate inf --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/tau${TAU}_c32.log 2>/dev/null
  echo "tau=$TAU C32: $(grep -h 'Output token throughput' $B/tau${TAU}_c32.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/tau${TAU}_c32.log|awk '{print $NF}')" | tee -a $B/RUN.log
  python3 $SP/gsm_eval.py $B/tau${TAU}_gsm40.json 40 2>&1 | tail -1 | sed "s/^/tau=$TAU GSM40: /" | tee -a $B/RUN.log
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; echo "TAU_SWEEP_DONE $B"
