#!/usr/bin/env bash
# PLN_008 M1 v2: ctx=2000 은 random 데이터셋(sonnet 긴프롬프트 생성 행업 회피), 벤치는 컨테이너 내부 파일 기록 + in-container timeout.
# 미측정 셀만: 인자로 셀 인덱스 목록 (기본: 0 1 2 3 5 6 7 8 10 11)
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
REPO=$HOME/projects/vllm_hybrid
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
BASE=$REPO/eval/results/20260908_124345_pln008_m1_cells   # 기존 디렉토리에 append
mkdir -p $BASE; RUN=$BASE/RUN_v2.log
FILTER="${*:-0 1 2 3 5 6 7 8 10 11}"
echo "== M1 v2 $(date +%H:%M:%S) cells=[$FILTER] ==" | tee -a $RUN
mapfile -t CELLS < <(python3 -c '
import json
p=json.load(open("/home/mystous/projects/vllm_hybrid/shadow_assists/features/IDE_031/predictions.json"))
for i,x in enumerate(p["predictions"]):
    c=x["cell"]; print(i, c["H"], c["C"], c["ctx"], c["kv"], x["kv_tokens"])
')
boot() { local H=$1 KV=$2 KVT=$3 extra=""; [ "$KV" = hicache ] && extra="--enable-hierarchical-cache --hicache-size 64"
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens $KVT --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $H --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 4 $extra > /tmp/sgl_m1v2_h${H}_${KV}.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "  boot H=$H kv=$KV ok ${i}s" | tee -a $RUN; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "  BOOT_FAIL H=$H kv=$KV" | tee -a $RUN; return 1; }; sleep 10; i=$((i+10)); done; return 1; }
bench() { local idx=$1 C=$2 CTX=$3 rep=$4 ds args np
  if [ "$CTX" = 2000 ]; then ds="--dataset-name random --random-input-len 1600 --random-output-len 512 --random-prefix-len 1500"; np=$((C<8?8:C));
  else ds="--dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100"; np=$((C*4<32?32:C*4)); fi
  local rf=/tmp/m1cell_${idx}_r${rep}.log
  docker exec -d vllm-h100 bash -c "timeout 400 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK $ds --num-prompts $np --max-concurrency $C --request-rate inf --seed $((42+rep)) --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>460)) && break; done
  docker cp vllm-h100:$rf $BASE/cell${idx}_C${C}_ctx${CTX}_r${rep}.log 2>/dev/null
  local tpot=$(grep -h 'Mean TPOT' $BASE/cell${idx}_C${C}_ctx${CTX}_r${rep}.log | awk '{print $NF}')
  local tput=$(grep -h 'Output token throughput' $BASE/cell${idx}_C${C}_ctx${CTX}_r${rep}.log | awk '{print $NF}')
  echo "cell $idx C=$C ctx=$CTX rep=$rep TPOT=${tpot:-NA} tput=${tput:-NA}" | tee -a $RUN; }
for cfg in $(for line in "${CELLS[@]}"; do set -- $line; grep -qw "$1" <<<"$FILTER" && echo "$2_$5_$6"; done | sort -u); do
  H=${cfg%%_*}; rest=${cfg#*_}; KV=${rest%%_*}; KVT=${rest#*_}
  boot $H $KV $KVT || continue
  for line in "${CELLS[@]}"; do set -- $line; [ "$2_$5_$6" = "$cfg" ] || continue; grep -qw "$1" <<<"$FILTER" || continue
    for rep in 1 2; do bench $1 $3 $4 $rep; done; done
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "== M1 v2 DONE $(date +%H:%M:%S) ==" | tee -a $RUN
