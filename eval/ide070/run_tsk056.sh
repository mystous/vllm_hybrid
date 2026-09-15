#!/usr/bin/env bash
# IDE_070 / TSK_056 — layer 별 비균일 GPU expert 배치 (지시서 C 시리즈). 총 슬롯 5,952 = hot96 과 동일 HBM, KV 40,960 고정.
#   u96v2  : uniform 96 + hotmap_v2 (측정 트레이스로 재생성) — hotmap 갱신 효과 분리
#   nu5952 : 층별 예산 (layer_budget_5952.json, 75~142) + hotmap_v2 + kt_ep_wrapper 패치 (KT_GPU_EXPERTS_PER_LAYER)
#   (기준선 u96v1 = IDE_069 hotmap, run_measure*.sh 의 값 사용)
# 사용: run_tsk056.sh <cells: u96v2,nu5952>   REPS=3 C=64
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
CELLS=${1:-u96v2,nu5952}; REPS=${REPS:-3}; C=${C:-64}; N=$((C*4))
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_tsk056
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480; PATCH=$REPO/eval/ide070/patch_per_layer_experts.sh
args() {  # args <hotmap_in_container>
  echo "--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location $1 --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic"
}
run_cell() {  # run_cell <name> <hotmap> <BOOT_ENV>
  local name=$1 hm=$2 out=$BASE/$1; export BOOT_ENV=$3
  log "-- cell $name hotmap=$hm env='$BOOT_ENV' --"
  boot "$out" 0,1,2,3 2400 $(args "$hm")
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    docker exec "$CN" bash -c "grep -c 'IDE_070 per-layer' $LOG_IN_CN; grep 'IDE_070 per-layer' $LOG_IN_CN | head -3" > "$out/patch_log.txt" 2>&1
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/hbm_after_boot.csv"
    smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    bench "$out/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1
    for r in $(seq 1 $REPS); do
      bench "$out/rep$r" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
      cpu_summary "$out/rep$r/cpu_util.txt" | tee -a "$RUN_LOG"
    done
    docker exec "$CN" bash -c "tail -60 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  else
    log "cell $name boot failed: $(cat "$out/error_lines.log" | tail -3)"
  fi
  stop_server; unset BOOT_ENV
}
log "== IDE_070 TSK_056 $TS cells=$CELLS reps=$REPS C=$C =="
stop_server
for c in ${CELLS//,/ }; do
  case $c in
    u96v2)  run_cell u96v2 /models/kt/ide070/hotmap_v2.json "" ;;
    nu5952) bash "$PATCH" "$CN" | tee -a "$RUN_LOG"
            run_cell nu5952 /models/kt/ide070/hotmap_v2.json "KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json"
            bash "$PATCH" "$CN" --revert | tee -a "$RUN_LOG" ;;
    u96v1)  run_cell u96v1 /models/kt/ide069/hotmap.json "" ;;
    *) log "unknown cell $c" ;;
  esac
done
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
for d in $BASE/*/; do n=$(basename $d); for r in $d/rep*/; do [ -f $r/bench_summary.txt ] && log "$n $(basename $r): $(grep -E 'Output token|Median TPOT|P95 TPOT' $r/bench_summary.txt | tr '\n' ' ')"; done; done
log "== TSK_056 done =="
echo "$BASE"
