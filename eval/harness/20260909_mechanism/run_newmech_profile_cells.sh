#!/usr/bin/env bash
# (a) 새 메커니즘 (CF+skip+pin, α=0.25 hotmap) hot-96 C32 decode 프로파일 → 비-층 고정비 측정  (b) 사전등록 플래그 대기 → 12셀 측정
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "V3CELLS_DONE" $SP/v3cells.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
COMMON="--served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8"
ENVS="KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 CUDA_VISIBLE_DEVICES=0,1,2,3"
declare -A KVT=( [80]=131072 [96]=49152 ); declare -A HM=( [prompt]=/tmp/hotmap.json [alpha0.25]=/tmp/hotmap_mixed_0.25.json )
boot() { # tag H hotmap
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; rm -rf /tmp/sgl_prof5; mkdir -p /tmp/sgl_prof5; true'; sleep 5
  docker exec -d sgl-kt bash -c "$ENVS python3 -m sglang.launch_server --model-path $M $COMMON --max-total-tokens ${KVT[$2]} --kt-num-gpu-experts $2 --init-expert-location ${HM[$3]} > /tmp/sgl_$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; return 1; }
bench() { # tag C nprompts
  local rf=/tmp/nm_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $3 --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2: tok/s $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}') TTFT $(grep -h 'Mean TTFT' $B/$1.log|awk '{print $NF}')" | tee -a $B/RUN.log; }
# (a) 프로파일
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide033_newmech_profile; mkdir -p $B
boot prof 96 alpha0.25 && {
  rf=/tmp/nm_prof_bench.log; docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 160 --max-concurrency 32 --request-rate inf --seed 42 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  sleep 60; curl -s -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' -d '{"output_dir":"/tmp/sgl_prof5","num_steps":8,"activities":["CPU","GPU"]}' >/dev/null; sleep 20; curl -s -X POST http://127.0.0.1:30000/stop_profile >/dev/null
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>500)) && break; done
  docker cp sgl-kt:/tmp/sgl_prof5/. $B/ 2>/dev/null; python3 $SP/analyze_trace.py $B 2>&1 | tail -10 | tee -a $B/RUN.log; }
# prefill 법칙 재측정 (새 메커니즘, α=0.25, hot-96): 입력 512 / 출력 1, C=1/8/32 → 요청당 prefill ms
for c in 1 8 32; do rf=/tmp/nm_pf$c.log
  docker exec -d vllm-h100 bash -c "timeout 400 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 1 --sonnet-prefix-len 100 --num-prompts $(( c*8 > 64 ? 64 : c*8 )) --max-concurrency $c --request-rate inf --seed 42 --percentile-metrics ttft --metric-percentiles 50 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 5; w=$((w+5)); ((w>400)) && break; done; docker cp vllm-h100:$rf $B/prefill_c$c.log 2>/dev/null
  echo "prefill out=1 C$c: mean TTFT $(grep -h 'Mean TTFT' $B/prefill_c$c.log|awk '{print $NF}') ms, req/s $(grep -h 'Request throughput' $B/prefill_c$c.log|awk '{print $NF}')" | tee -a $B/RUN.log; done
echo "NEWMECH_PROFILE_DONE $B"
# (b) 사전등록 대기 → 12셀
until [ -f $SP/prereg_v3cells.flag ]; do sleep 30; done
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_pln008_newmech_cells; mkdir -p $B; cp $REPO/shadow_assists/features/IDE_031/predictions_newmech.json $B/ 2>/dev/null
for cfg in "96 prompt 8 64;16 96" "96 alpha0.25 8 64;16 96" "80 prompt 8 64;16 96;32 128;64 192" "80 alpha0.25 8 64;16 96;32 128;64 192"; do
  set -- $cfg; H=$1; HMK=$2; shift 2; cells="$*"
  boot H${H}_${HMK} $H $HMK || continue
  IFS=';' read -ra CL <<< "$cells"; for c in "${CL[@]}"; do set -- $c; bench H${H}_${HMK}_C$1 $1 $2; done
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "NEWMECH_CELLS_DONE $B"
