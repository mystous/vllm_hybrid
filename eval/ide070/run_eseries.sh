#!/usr/bin/env bash
# IDE_070 / TSK_059 — E 시리즈: GPU expert W4 (AWQ 4-bit g128 체크포인트 QuantTrio/…-AWQ, 252 GB) 로 HBM 당 expert 수 확대.
#   CPU expert 는 그대로 kt INT4 (FP8 스냅샷에서 변환) → attention·GPU expert 만 AWQ. 부팅 실패 = 스킵(기록).
#   e1_awq_h96  : hot 96 (기준선과 슬롯 수 동일 — W4 자체 효과)
#   e2_awq_h128 : hot 128
#   e3_awq_h144 : hot 144
#   KV 40,960, N=4, graph bs64, hotmap v1 (IDE_069)
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
CELLS=${1:-e1_awq_h96,e2_awq_h128,e3_awq_h144}; REPS=${REPS:-3}; C=64; N=$((C*4))
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_eseries
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
AWQ_HOST=$(ls -d $HOME/.cache/huggingface/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/*/ 2>/dev/null | head -1)
[ -n "$AWQ_HOST" ] || { log "AWQ snapshot not found — E series skipped"; exit 1; }
M_CN=/models/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/$(basename "$AWQ_HOST")
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480awq
args() {  # args <hot>
  echo "--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $1 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic"
}
run_cell() {  # run_cell <name> <hot> [extra args]
  local name=$1 hot=$2; shift 2; local out=$BASE/$name
  log "-- cell $name hot=$hot model=$M_CN extra='$*' --"
  boot "$out" 0,1,2,3 3000 $(args $hot) "$@"
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/hbm_after_boot.csv"
    smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    bench "$out/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1
    for r in $(seq 1 $REPS); do
      bench "$out/rep$r" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
      cpu_summary "$out/rep$r/cpu_util.txt" | tee -a "$RUN_LOG"
    done
    docker exec "$CN" bash -c "tail -60 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  else
    log "cell $name boot failed ($BOOT_VERDICT): $(tail -3 "$out/error_lines.log")"
    docker exec "$CN" bash -c "tail -400 $LOG_IN_CN" > "$out/server_boot_fail.log" 2>/dev/null
  fi
  stop_server
}
log "== IDE_070 E-series $TS cells=$CELLS reps=$REPS awq=$M_CN =="
stop_server
for c in ${CELLS//,/ }; do
  case $c in
    e1_awq_h96)  run_cell e1_awq_h96 96 ;;
    e2_awq_h128) run_cell e2_awq_h128 128 ;;
    e3_awq_h144) run_cell e3_awq_h144 144 ;;
    e1_awq_h96_nograph)  run_cell e1_awq_h96_nograph 96 --disable-cuda-graph ;;    # 그래프 캡처 단계 실패 → eager 재시도
    e2_awq_h128_nograph) run_cell e2_awq_h128_nograph 128 --disable-cuda-graph ;;
    *) log "unknown cell $c" ;;
  esac
done
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
for d in $BASE/*/; do n=$(basename $d); for r in $d/rep*/; do [ -f $r/bench_summary.txt ] && log "$n $(basename $r): $(grep -E 'Output token|Median TPOT|P95 TPOT' $r/bench_summary.txt | tr '\n' ' ')"; done; done
log "== E-series done =="
echo "$BASE"
