#!/usr/bin/env bash
# IDE_069 품질 게이트 — deferral 이 근사 연산이므로 GSM8K 40문항으로 deferral 0 대 N 을 대조한다.
# 사용: run_gsm_gate.sh <gpu_experts> <deferral_list(,)> [n=40]
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
GX=${1:-96}; DEFS=${2:-0,4}; N=${3:-40}
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide069_gsm${N}_hot${GX}
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $GX --init-expert-location /models/kt/ide069/hotmap.json --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic"
log "== IDE_069 GSM$N gate hot$GX defs=$DEFS =="
stop_server
for d in ${DEFS//,/ }; do
  out=$BASE/def$d; mkdir -p "$out"
  DEF=""; [ "$d" != 0 ] && DEF="--kt-max-deferred-experts-per-token $d"
  log "-- def$d --"
  boot "$out" 0,1,2,3 2400 $COMMON $DEF
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    t0=$(date +%s)
    "$HOME/venv-bench/bin/python" "$REPO/eval/harness/gsm_eval.py" "$out/gsm$N.json" "$N" "$MODEL" 2>&1 | tail -3 | tee -a "$RUN_LOG"
    log "gsm 소요 $(( $(date +%s) - t0 ))s"
  fi
  stop_server
done
log "== gate done =="
