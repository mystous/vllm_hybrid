#!/usr/bin/env bash
# IDE_046 서빙 A/B: 새 빌드에서 env off (base) vs KT_QA_PAR=1 KT_AMX_BCACHE=1 — C64 (TTFT 관찰), C160. IDE_046-b 종료 후.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE046B_DONE" $SP/ide046b.log 2>/dev/null; do sleep 20; done
grep -q "BUILD_FAIL" $SP/ide046b.log && { echo "skip: build fail"; echo "IDE046S_DONE"; exit 0; }
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$(ls -d $REPO/eval/results/*_ide046_amx_bcache | tail -1); echo "== serving A/B $(date +%H:%M)" | tee -a $B/RUN.log
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8"
boot() { docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$3 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $2 > /tmp/sgl_046s_$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s $(docker exec sgl-kt grep -m1 -o 'KV Cache is allocated.*' /tmp/sgl_046s_$1.log | cut -c1-90)" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL $(docker exec sgl-kt grep -m1 -o 'OOM on device.*' /tmp/sgl_046s_$1.log | cut -c1-120)" | tee -a $B/RUN.log; return 1; }
bench() { local rf=/tmp/i46s_$1.log; local NP=$(( $2*3 ))
  docker exec -d vllm-h100 bash -c "timeout 600 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>660)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2: $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}'), TTFT $(grep -h 'Mean TTFT' $B/$1.log|awk '{print $NF}')  $3" | tee -a $B/RUN.log; }
ok=0
KVARGS="--kv-cache-dtype fp8_e5m2 --mem-fraction-static 0.94 --chunked-prefill-size 8192 --max-total-tokens 110592 --cuda-graph-max-bs 160"
boot s046_on "$KVARGS" "KT_QA_PAR=1 KT_AMX_BCACHE=1" && { ok=1
  bench on_c64_r1 64 "(기준 766/60.6/TTFT 2.99s)"; bench on_c64_r2 64 "(r2)"; bench on_c160 160 "(기준 965~985/120)"
  python3 $SP/gsm_eval.py $B/gsm40_on.json 40 2>&1 | tail -1 | sed "s/^/on GSM40: /" | tee -a $B/RUN.log; }
boot s046_off "$KVARGS" "" && { ok=1
  bench off_c64_r1 64 "(대조)"; bench off_c64_r2 64 "(r2)"; bench off_c160 160 "(대조)"; }
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "IDE046S_DONE $B"
