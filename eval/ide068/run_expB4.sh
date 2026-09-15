#!/usr/bin/env bash
# IDE_068 / TSK_050 확장 B4 — GPU 3·5·6·7 장: TP 가 성립하는 조합 탐색 (사용자 요청).
# 480B: attention heads 96, KV heads 8. TP 가 head 수를 나누지 못하면 엔진이 거부한다.
# 추측으로 제외하지 않고 네 경우 모두 부팅을 시도한다. 되면 greedy + 벤치, 안 되면 오류 줄을 기록.
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide068_expB4_480b_tp3567
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --attention-backend triton --trust-remote-code"
KT="--disable-cuda-graph --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 0"
log "== IDE_068 expB4 start $TS =="
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
  else
    log "실패 사유:"; grep -vE "Ignore import error" "$out/error_lines.log" | tail -3 | cut -c1-200 | tee -a "$RUN_LOG"
    docker exec "$CN" bash -c "grep -nE 'Error|error|assert|divisible|must be' $LOG_IN_CN | grep -v 'Ignore import' | tail -8" > "$out/error_detail.log" 2>/dev/null
  fi
  stop_server
}
# 하이브리드 (expert 전량 CPU) — TP3 / TP5 / TP6 / TP7
run_cell b9_hyb_tp3  0,1,2           900 $COMMON --tp 3 --mem-fraction-static 0.80 --max-total-tokens 65536 $KT
run_cell b10_hyb_tp5 0,1,2,3,4       900 $COMMON --tp 5 --mem-fraction-static 0.80 --max-total-tokens 65536 $KT
run_cell b11_hyb_tp6 0,1,2,3,4,5     900 $COMMON --tp 6 --mem-fraction-static 0.80 --max-total-tokens 65536 $KT
run_cell b12_hyb_tp7 0,1,2,3,4,5,6   900 $COMMON --tp 7 --mem-fraction-static 0.80 --max-total-tokens 65536 $KT
# GPU-only 도 같은 장수에서 시도 (분할 가능성·용량을 함께 확인)
run_cell b13_gpu_tp6_ep6 0,1,2,3,4,5   900 $COMMON --tp 6 --ep-size 6 --mem-fraction-static 0.90 --max-total-tokens 65536
run_cell b14_gpu_tp7_ep7 0,1,2,3,4,5,6 900 $COMMON --tp 7 --ep-size 7 --mem-fraction-static 0.90 --max-total-tokens 65536
log "== expB4 done $(date +%H:%M:%S) =="
