#!/usr/bin/env bash
# IDE_034 3차: phase-가중 hot set α 스윕 (α = prefill 가중치). 2차 종료 후. hot-96 N=8 CF+skip+pin, C32 (TTFT/TPOT/tput) → 최선 α 로 C64+GSM40 → v3 재개
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "DECODE2_DONE" $SP/decode_hotmap_r2.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide034_mixed_hotmap; mkdir -p $B
docker cp $SP/build_hotmap_mixed.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/build_hotmap_mixed.py 0.25 0.5 0.75 2>&1 | tee -a $B/RUN.log
for a in 0.25 0.5 0.75; do docker cp sgl-kt:/tmp/hotmap_mixed_$a.json $B/ 2>/dev/null; done
COMMON="--served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8"
ENVS="KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_TQ_TIMING=1 CUDA_VISIBLE_DEVICES=0,1,2,3"
boot() { docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "$ENVS python3 -m sglang.launch_server --model-path $M $COMMON --init-expert-location $2 > /tmp/sgl_$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; return 1; }
bench() { local rf=/tmp/mx_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $3 --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2: $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}'), TTFT $(grep -h 'Mean TTFT' $B/$1.log|awk '{print $NF}')  $4" | tee -a $B/RUN.log; }
best=""; bestv=0
for a in 0.5 0.25 0.75; do
  boot mx$a /tmp/hotmap_mixed_$a.json || continue
  bench mx${a}_c32 32 128 "(prompt 600/43.5/1304, decode-only 554/38.0/2560)"
  v=$(grep -h 'Output token throughput' $B/mx${a}_c32.log | awk '{print $NF}'); if python3 -c "import sys; sys.exit(0 if float('$v' or 0) > float('$bestv') else 1)"; then bestv=$v; best=$a; fi
done
echo "best alpha=$best ($bestv tok/s)" | tee -a $B/RUN.log
if [ -n "$best" ]; then boot mxbest /tmp/hotmap_mixed_$best.json && { bench mxbest_c64 64 192 "(prompt C64 760~786)"; python3 $SP/gsm_eval.py $B/gsm40.json 40 2>&1 | tail -1 | sed "s/^/MIXED($best) GSM40: /" | tee -a $B/RUN.log; }; fi
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "MIXED_DONE $B"
