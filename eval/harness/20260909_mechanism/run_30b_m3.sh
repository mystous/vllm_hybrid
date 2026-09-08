#!/usr/bin/env bash
# M3 30B 반례: Qwen3-30B-A3B-FP8, TP=1 (GPU0). GPU-only vs hybrid hot-96 (kt INT4, CF+skip+pin, identity hot set). C32/C64. 바닥 프로브 종료 후.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "FLOOR_PROBE_DONE" $SP/floor_probe.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_pln008_m3_30b; mkdir -p $B
COMMON="--served-model-name q30 --host 127.0.0.1 --port 30000 --tp 1 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --cuda-graph-max-bs 64 --mem-fraction-static 0.90 --max-total-tokens 131072"
run() { # tag args envs
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$3 python3 -m sglang.launch_server --model-path $M $COMMON $2 > /tmp/sgl_30b_$1.log 2>&1"
  local i=0 ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  [ $ok = 1 ] || { echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A10 -E "Traceback|Error" /tmp/sgl_30b_$1.log | tail -20 >> $B/RUN.log; return 1; }
  echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
  for C in 32 64; do local rf=/tmp/m30_$1_c$C.log; NP=$(( C==32 ? 128 : 192 ))
    docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q30 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $C --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
    local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1_c$C.log 2>/dev/null
    echo "$1 C$C: tok/s $(grep -h 'Output token throughput' $B/$1_c$C.log|awk '{print $NF}') TPOT $(grep -h 'Mean TPOT' $B/$1_c$C.log|awk '{print $NF}') TTFT $(grep -h 'Mean TTFT' $B/$1_c$C.log|awk '{print $NF}')" | tee -a $B/RUN.log; done; }
run gpuonly "" "CUDA_VISIBLE_DEVICES=0"
run hybrid96 "--kt-weight-path /models/kt/qwen3-30b-int4-b --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --kt-num-gpu-experts 96" "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 CUDA_VISIBLE_DEVICES=0"
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "M3_30B_DONE $B"
