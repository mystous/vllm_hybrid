#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
REPO=$HOME/projects/vllm_hybrid
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_pln008_m1v2_cells; mkdir -p $B
cp $REPO/shadow_assists/features/IDE_031/predictions_v2.json $B/
declare -A IO=( [1000]="1000 256 900" [3000]="3000 256 2900" )
boot() { docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 131072 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $1 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 4 > /tmp/sgl_v2_h$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "boot H=$1 ok" | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "BOOT_FAIL H=$1" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done; return 1; }
bench() { local H=$1 C=$2 ctx=$3 rep=$4; set -- ${IO[$ctx]}; local IN=$1 OUT=$2 PRE=$3
  local rf=/tmp/v2_${H}_${C}_${ctx}_r${rep}.log
  docker exec -d vllm-h100 bash -c "timeout 400 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name random --random-input-len $IN --random-output-len $OUT --random-prefix-len $PRE --num-prompts $((C<8?8:C)) --max-concurrency $C --request-rate inf --seed $((42+rep)) --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>460)) && break; done
  docker cp vllm-h100:$rf $B/H${H}_C${C}_ctx${ctx}_r${rep}.log 2>/dev/null
  echo "H$H C$C ctx$ctx r$rep TPOT=$(grep -h 'Mean TPOT' $B/H${H}_C${C}_ctx${ctx}_r${rep}.log|awk '{print $NF}')" | tee -a $B/RUN.log; }
for H in 80 96; do boot $H || continue
  for ctx in 1000 3000; do for C in 16 32; do for rep in 1 2; do bench $H $C $ctx $rep; done; done; done
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "V2CELLS_DONE $B"
