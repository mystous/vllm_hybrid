#!/usr/bin/env bash
# IDE_068 / TSK_050 후속 B3 — 최소 장수(TP1)에서 cuda graph 를 켠 현실적 구성.
# b3/b4 는 TSK_047 조건(--disable-cuda-graph)이었다. IDE_030 은 KT 경로에서
# `--cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64` 로 graph 가 동작함을 보였다.
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide068_expB3_480b_tp1_cudagraph
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --attention-backend triton --trust-remote-code"
KTG="--cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2"
log "== IDE_068 expB3 start $TS =="
stop_server
run_cell() {
  local name=$1 gpus=$2 tmo=$3; shift 3
  local out=$BASE/$name
  log "-- cell $name --"
  boot "$out" "$gpus" "$tmo" "$@"
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    log "smoke:"; smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    log "bench C16 n=64:"; bench "$out" "$MODEL" "$TOK" 16 64 | tee -a "$RUN_LOG"
    cpu_summary "$out/cpu_util.txt" | tee -a "$RUN_LOG"
    docker exec "$CN" bash -c "tail -150 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  fi
  stop_server
}
run_cell b7_hyb_tp1_gx0_graph  0 3600 $COMMON --tp 1 --mem-fraction-static 0.80 --max-total-tokens 24576 $KTG --kt-num-gpu-experts 0
run_cell b8_hyb_tp1_gx16_graph 0 3600 $COMMON --tp 1 --mem-fraction-static 0.88 --max-total-tokens 24576 $KTG --kt-num-gpu-experts 16
log "== expB3 done $(date +%H:%M:%S) =="
