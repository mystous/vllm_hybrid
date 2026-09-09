#!/usr/bin/env bash
# 최종 구성 (운영점 B + RB/PF/FUSE) 정상 상태 도착률: delayer 끔 vs 켬(게이트 128), rate 5/7/9 + 버스트 C224
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide052_mixed_chunk; mkdir -p $B
declare -A CFGS=( [mixed]="--enable-mixed-chunk" [mixed_delay225]="--enable-mixed-chunk --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.10 --prefill-delayer-max-delay-passes 8 --prefill-delayer-max-delay-ms 500" )
declare -A GATE=( [mixed]=225 [mixed_delay225]=225 )
for CFG in mixed mixed_delay225; do EXTRA=${CFGS[$CFG]}; export PDGATE=${GATE[$CFG]}
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
docker exec -d sgl-kt bash -c "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 SGL_PD_MAX_RUNNING_FOR_DELAY=$PDGATE KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --kv-cache-dtype fp8_e5m2 --mem-fraction-static 0.94 --max-total-tokens 143360 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224 --chunked-prefill-size 4096 $EXTRA > /tmp/sgl_052_$CFG.log 2>&1"
i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
[ $ok = 1 ] || { echo "$CFG BOOT_FAIL $(docker exec sgl-kt grep -m1 -E "Error|error" /tmp/sgl_052_$CFG.log | cut -c1-150)" | tee -a $B/RUN.log; continue; }
echo "$CFG boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
for RATE in 5 7 9; do rf=/tmp/i52_${CFG}_r$RATE.log
  docker exec -d vllm-h100 bash -c "timeout 900 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 480 --max-concurrency 224 --request-rate $RATE --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>960)) && break; done; docker cp vllm-h100:$rf $B/${CFG}_rate$RATE.log 2>/dev/null
  echo "$CFG rate $RATE req/s: tok/s $(grep -h 'Output token throughput' $B/${CFG}_rate$RATE.log|awk '{print $NF}'), TPOT $(grep -h 'Mean TPOT' $B/${CFG}_rate$RATE.log|awk '{print $NF}'), TTFT p50 $(grep -h 'Median TTFT' $B/${CFG}_rate$RATE.log|awk '{print $NF}') p99 $(grep -h 'P99 TTFT' $B/${CFG}_rate$RATE.log|awk '{print $NF}')" | tee -a $B/RUN.log; done
  rf=/tmp/i52_${CFG}_burst.log; docker exec -d vllm-h100 bash -c "timeout 900 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 672 --max-concurrency 224 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>960)) && break; done; docker cp vllm-h100:$rf $B/${CFG}_burst.log 2>/dev/null
  [ $CFG = mixed ] && python3 $SP/gsm_eval.py $B/gsm40_$CFG.json 40 2>&1 | tail -1 | sed "s/^/$CFG GSM40: /" | tee -a $B/RUN.log
  echo "$CFG burst C224: tok/s $(grep -h 'Output token throughput' $B/${CFG}_burst.log|awk '{print $NF}'), TPOT $(grep -h 'Mean TPOT' $B/${CFG}_burst.log|awk '{print $NF}'), TTFT p50 $(grep -h 'Median TTFT' $B/${CFG}_burst.log|awk '{print $NF}')  (기준 1098.2/149.6)" | tee -a $B/RUN.log
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "IDE052_DONE $B"
