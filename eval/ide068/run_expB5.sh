#!/usr/bin/env bash
# IDE_068 확장 B5 — TP 로 쓸 수 없는 장수(3·5·6·7)를 DP(복제)로 쓰는 경우의 데이터 1점.
# TP3/5/6/7 은 vocab 151,936 이 나뉘지 않아 엔진이 거부한다 (B4). DP 는 rank 마다 모델
# 전체(TP1)를 복제하므로 장수 제약이 없다. 단 rank 마다 CPU expert 풀이 생기므로
# DRAM 은 rank 수 배(232 GB × k), cpuinfer 는 rank 당 96/k 로 나눠 총 96 을 유지한다.
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide068_expB5_480b_dp
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --attention-backend triton --trust-remote-code --disable-cuda-graph --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-num-gpu-experts 0"
log "== IDE_068 expB5 start $TS =="
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
    log "bench C48 n=96:"; mkdir -p "$out/c48"; bench "$out/c48" "$MODEL" "$TOK" 48 96 | tee -a "$RUN_LOG"
    cpu_summary "$out/cpu_util.txt" | tee -a "$RUN_LOG"
    docker exec "$CN" bash -c "tail -150 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  else
    log "실패 사유:"; grep -vE "Ignore import error" "$out/error_lines.log" | tail -3 | cut -c1-200 | tee -a "$RUN_LOG"
    docker exec "$CN" bash -c "grep -nE 'Error|error|assert' $LOG_IN_CN | grep -v 'Ignore import' | tail -8" > "$out/error_detail.log" 2>/dev/null
  fi
  stop_server
}
# DP3 × TP1 — GPU 3장. rank 당 cpuinfer 32 (총 96). DRAM 예상 232×3 ≈ 700 GB.
run_cell b15_hyb_dp3_tp1 0,1,2 2400 $COMMON --tp 1 --dp-size 3 --kt-cpuinfer 32 --mem-fraction-static 0.80 --max-total-tokens 24576
log "== expB5 done $(date +%H:%M:%S) =="
