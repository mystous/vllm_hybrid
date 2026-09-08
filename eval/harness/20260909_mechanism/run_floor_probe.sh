#!/usr/bin/env bash
# 핸드오프 바닥 판별: 235B hot-128 (cold 0) 하이브리드 경로 vs GPU-only. (a) CF+skip_imm+skip_def (b) host 콜백 N=8. C32 각각. CF+τ 종료 후.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "CFTAU_DONE" $SP/cf_tau.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide036_floor_probe; mkdir -p $B
KT="--mem-fraction-static 0.92 --max-total-tokens 65536 --kt-weight-path /models/kt/qwen3-235b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --kt-num-gpu-experts 128 --init-expert-location /tmp/hm235/hotmap_alpha0.25.json"
run() { # tag envs
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$2 python3 -m sglang.launch_server --model-path $M --served-model-name q235 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --cuda-graph-max-bs 64 $KT > /tmp/sgl_floor_$1.log 2>&1"
  local i=0 ok=0; while ((i<1800)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  [ $ok = 1 ] || { echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A10 -E "Traceback|Error|OOM" /tmp/sgl_floor_$1.log | tail -20 >> $B/RUN.log; return 1; }
  echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
  local rf=/tmp/floor_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q235 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 128 --max-concurrency 32 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 hot-128 C32: tok/s $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}')  (GPU-only 930/21.2; hybrid hot-96 α 804/28.7)" | tee -a $B/RUN.log; }
run cf_skipboth "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_CF_SKIP_EMPTY_DEF=1 CUDA_VISIBLE_DEVICES=0,1,2,3"
run hostcb "CUDA_VISIBLE_DEVICES=0,1,2,3"
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "FLOOR_PROBE_DONE $B"
