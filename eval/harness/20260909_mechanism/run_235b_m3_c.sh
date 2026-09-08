#!/usr/bin/env bash
# M3 235B: (a) 하이브리드 hot-96 (identity hot set) + recorder 트레이스 → hotmap prompt/α.25 + 통계  (b) 사전등록 플래그 대기 → 측정:
#   GPU-only (kt 없음) C32/C64/GSM40 | hybrid hot-96 prompt-map (CF+skip+pin) C32/C64 | hybrid hot-96 α.25 C32/C64/GSM40 | hybrid hot-64 α.25 C64 (KV 여유)
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "GSMATTR_DONE" $SP/gsm_attr.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_pln008_m3_235b; mkdir -p $B; docker cp $SP/pin_nonkt.sh sgl-kt:/tmp/pin_nonkt.sh; docker cp $SP/build_hotmaps_generic.py sgl-kt:/tmp/
COMMON="--served-model-name q235 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --cuda-graph-max-bs 64"
KT="--kt-weight-path /models/kt/qwen3-235b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8"
ENVS="KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 CUDA_VISIBLE_DEVICES=0,1,2,3"
boot() { # tag args envs
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$3 python3 -m sglang.launch_server --model-path $M $COMMON $2 > /tmp/sgl_235_$1.log 2>&1"
  local i=0; while ((i<1800)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A10 -E "Traceback|Error|OOM|out of memory" /tmp/sgl_235_$1.log | tail -24 >> $B/RUN.log; return 1; }
bench() { # tag C nprompts
  local rf=/tmp/m3_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q235 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $3 --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2: tok/s $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}') TTFT $(grep -h 'Mean TTFT' $B/$1.log|awk '{print $NF}')" | tee -a $B/RUN.log; }
gsm() { python3 $SP/gsm_eval.py $B/gsm40_$1.json 40 q235 2>&1 | tail -1 | sed "s/^/$1 GSM40: /" | tee -a $B/RUN.log; }
# (a) 트레이스
docker exec sgl-kt bash -c 'rm -f /tmp/expert_distribution_recorder_*.pt; true'
boot trace "--mem-fraction-static 0.90 --max-total-tokens 65536 $KT --kt-num-gpu-experts 96 --expert-distribution-recorder-mode per_pass" "$ENVS" || { echo "M3_235B_DONE"; exit 1; }
for P in "The capital of France is" "def fibonacci(n):\n    "; do curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"q235\",\"prompt\":\"$P\",\"max_tokens\":24,\"temperature\":0}" | python3 -c "import json,sys; print('  greedy:', repr(json.load(sys.stdin)['choices'][0]['text'][:70]))" | tee -a $B/RUN.log; done
curl -s -X POST http://127.0.0.1:30000/start_expert_distribution_record >/dev/null
bench trace_c32 32 128
curl -s -X POST http://127.0.0.1:30000/stop_expert_distribution_record >/dev/null; curl -s -X POST http://127.0.0.1:30000/dump_expert_distribution_record >/dev/null; sleep 5
docker exec sgl-kt python3 /tmp/build_hotmaps_generic.py identity /tmp/hm235 0.25 0.5 2>&1 | tee -a $B/RUN.log
docker cp sgl-kt:/tmp/hm235 $B/hm235 2>/dev/null; cp $B/hm235/routing_phase_stats.json $REPO/shadow_assists/features/IDE_031/routing_phase_stats_235b.json 2>/dev/null
echo "M3_235B_TRACE_DONE $B"
# (b) 사전등록 대기
until [ -f $SP/prereg_235b.flag ]; do sleep 30; done
cp $REPO/shadow_assists/features/IDE_031/predictions_235b.json $B/ 2>/dev/null
boot gpuonly "--mem-fraction-static 0.92 --max-total-tokens 65536" "CUDA_VISIBLE_DEVICES=0,1,2,3" && { bench gpuonly_c32 32 128; bench gpuonly_c64 64 192; gsm gpuonly; }
boot h96_prompt "--mem-fraction-static 0.90 --max-total-tokens 131072 $KT --kt-num-gpu-experts 96 --init-expert-location /tmp/hm235/hotmap_prompt.json" "$ENVS" && { bench h96_prompt_c32 32 128; bench h96_prompt_c64 64 192; }
boot h96_a25 "--mem-fraction-static 0.90 --max-total-tokens 131072 $KT --kt-num-gpu-experts 96 --init-expert-location /tmp/hm235/hotmap_alpha0.25.json" "$ENVS" && { bench h96_a25_c32 32 128; bench h96_a25_c64 64 192; gsm h96_a25; }
boot h64_a25 "--mem-fraction-static 0.90 --max-total-tokens 131072 $KT --kt-num-gpu-experts 64 --init-expert-location /tmp/hm235/hotmap_alpha0.25.json" "$ENVS" && { bench h64_a25_c64 64 192; }
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "M3_235B_DONE $B"
