#!/usr/bin/env bash
# PLN_008 M1 측정 하네스: predictions.json 의 12셀을 (H, kv) 별로 묶어 부팅 1회당 여러 (C, ctx) 벤치. 셀당 2회 반복.
# 사용: run_m1_cells.sh [셀 인덱스 필터 e.g. "0 3 5"]  (미지정 시 전체)
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TS=$(date +%Y%m%d_%H%M%S); BASE=$REPO/eval/results/${TS}_pln008_m1_cells; mkdir -p $BASE
cp $REPO/shadow_assists/features/IDE_031/predictions.json $BASE/predictions_at_measure.json
echo "== M1 cells $TS ==" | tee $BASE/RUN.log

# 셀 목록: (idx H C ctx kv kv_tokens) — predictions.json 순서
mapfile -t CELLS < <(python3 - <<'EOF'
import json
p=json.load(open("/home/mystous/projects/vllm_hybrid/shadow_assists/features/IDE_031/predictions.json"))
for i,x in enumerate(p["predictions"]):
    c=x["cell"]; print(i, c["H"], c["C"], c["ctx"], c["kv"], x["kv_tokens"])
EOF
)
FILTER="${1:-}"
CUR_CFG=""
boot() { # H kv kv_tokens
  local H=$1 KV=$2 KVT=$3; local extra=""
  [ "$KV" = "hicache" ] && extra="--enable-hierarchical-cache --hicache-size 64"
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens $KVT --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $H --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 4 $extra > /tmp/sgl_m1_h${H}_${KV}.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "  boot H=$H kv=$KV kvt=$KVT ok ${i}s" | tee -a $BASE/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "  BOOT_FAIL H=$H kv=$KV" | tee -a $BASE/RUN.log; return 1; }; sleep 10; i=$((i+10)); done; return 1; }
bench() { # idx C ctx rep -> TPOT
  local idx=$1 C=$2 CTX=$3 rep=$4 IN OUT PRE NP
  if [ "$CTX" = "2000" ]; then IN=1600; OUT=512; PRE=1500; NP=$((C*3)); else IN=612; OUT=128; PRE=100; NP=$((C*4)); fi
  [ $NP -lt 32 ] && NP=32
  docker exec vllm-h100 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions \
    --model q480 --tokenizer "$TOK" --dataset-name sonnet --dataset-path /tmp/sonnet.txt \
    --sonnet-input-len $IN --sonnet-output-len $OUT --sonnet-prefix-len $PRE \
    --num-prompts $NP --max-concurrency $C --request-rate inf --seed $((42+rep)) \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $BASE/cell${idx}_C${C}_ctx${CTX}_r${rep}.log 2>&1
  local tpot=$(grep -h 'Mean TPOT' $BASE/cell${idx}_C${C}_ctx${CTX}_r${rep}.log | awk '{print $NF}')
  local tput=$(grep -h 'Output token throughput' $BASE/cell${idx}_C${C}_ctx${CTX}_r${rep}.log | awk '{print $NF}')
  echo "cell $idx C=$C ctx=$CTX rep=$rep TPOT=$tpot tput=$tput" | tee -a $BASE/RUN.log; }

# (H, kv, kvt) 별로 정렬해 부팅 횟수 최소화
for cfg in $(printf "%s\n" "${CELLS[@]}" | awk '{print $2"_"$5"_"$6}' | sort -u); do
  H=${cfg%%_*}; rest=${cfg#*_}; KV=${rest%%_*}; KVT=${rest#*_}
  todo=()
  for line in "${CELLS[@]}"; do set -- $line; [ "$2_$5_$6" = "$cfg" ] || continue; [ -n "$FILTER" ] && ! grep -qw "$1" <<<"$FILTER" && continue; todo+=("$1 $3 $4"); done
  [ ${#todo[@]} -eq 0 ] && continue
  boot $H $KV $KVT || continue
  curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d '{"model":"q480","prompt":"The capital of France is","max_tokens":6,"temperature":0}' | python3 -c "import json,sys; print('  Q1:', json.load(sys.stdin)['choices'][0]['text'][:40])" | tee -a $BASE/RUN.log
  for t in "${todo[@]}"; do set -- $t; for rep in 1 2; do bench $1 $2 $3 $rep; done; done
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "== M1 DONE $(date +%H:%M:%S) ==" | tee -a $BASE/RUN.log; echo "$BASE"
