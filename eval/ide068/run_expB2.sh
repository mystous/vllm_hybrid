#!/usr/bin/env bash
# IDE_068 / TSK_050 후속 — 최소 장수(TP1·TP2)에서 남는 HBM 에 expert 를 올릴 때의 처리량 회복.
# hotmap(빈도 기반 배치)은 IDE_030 산출물이 소실돼 없다. `--kt-num-gpu-experts N` 만 주면
# 물리 id 0..N-1 이 GPU 에 오르므로 이 결과는 hot-expert 이득의 **하한**이다.
# FP8 expert 1개/층 ≈ 3×6144×2560 B ≈ 47 MB → 62층 기준 N개 = N×2.9 GB (TP 로 나뉨).
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide068_expB2_480b_gpuexperts
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --attention-backend triton --trust-remote-code"
KTB="--disable-cuda-graph --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2"
log "== IDE_068 expB2 start $TS =="
stop_server

run_cell() {   # run_cell <name> <gpus> <timeout> <args...>
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

# TP1: 여유 ≈ 80 − 17 = 63 GB → 16개(46 GB) / 20개(58 GB)
run_cell b4_hyb_tp1_gx16 0 3600 $COMMON --tp 1 --mem-fraction-static 0.88 --max-total-tokens 24576 $KTB --kt-num-gpu-experts 16
# TP2: 여유 ≈ 63 GB × 2 → 40개(116 GB, 장당 58 GB)
run_cell b5_hyb_tp2_gx40 0,1 3600 $COMMON --tp 2 --mem-fraction-static 0.88 --max-total-tokens 32768 $KTB --kt-num-gpu-experts 40
# TP4: IDE_030 과 같은 96개 (장당 70 GB) — 참조. 단 hotmap 없음.
run_cell b6_hyb_tp4_gx96 0,1,2,3 3600 $COMMON --tp 4 --mem-fraction-static 0.92 --max-total-tokens 24576 $KTB --kt-num-gpu-experts 96

log "== expB2 done $(date +%H:%M:%S) =="
echo "$BASE"
