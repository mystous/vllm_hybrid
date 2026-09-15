#!/usr/bin/env bash
# IDE_070 / TSK_057 — B 시리즈 NUMA (코드 무변경). 기준선 hot96 def4 KV40k TP4, C64, 3회.
#   b1_interleave : numactl --interleave=all (가중치·버퍼를 두 노드에 교대 배치)
#   b2_membind0   : numactl --membind=0 (GPU0-3 소속 socket0 메모리만; socket1 스레드는 전부 원격)
#   b3_pool1      : --kt-threadpool-count 1 (NUMA 서브풀 없이 단일 풀)
#   b4_local      : numactl --cpunodebind=0 --membind=0 + --kt-cpuinfer 56 (socket0 전용, 국소 풀)
# 사용: run_bseries.sh <cells>
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
CELLS=${1:-b1_interleave,b2_membind0,b3_pool1,b4_local}; REPS=${REPS:-3}; C=64; N=$((C*4))
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_bseries
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
base_args() {  # base_args <cpuinfer> <threadpool_count>
  echo "--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer $1 --kt-threadpool-count $2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic"
}
run_cell() {  # run_cell <name> <BOOT_ENV prefix> <cpuinfer> <pool>
  local name=$1; export BOOT_ENV=$2; local ci=$3 pool=$4
  local out=$BASE/$name
  log "-- cell $name prefix='$BOOT_ENV' cpuinfer=$ci pool=$pool --"
  boot "$out" 0,1,2,3 2400 $(base_args $ci $pool)
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    TP0=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
    sudo numastat -p $TP0 > "$out/numastat_tp0.txt" 2>&1; numastat -m > "$out/numastat_m.txt" 2>&1
    for t in /proc/$TP0/task/*; do echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status | cut -f2)"; done | sort | uniq -c | sort -rn > "$out/tp0_thread_affinity.txt"
    smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    bench "$out/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1
    for r in $(seq 1 $REPS); do
      ( sudo pcm-numa 2 -csv="$out/rep$r.pcm_numa.csv" > /dev/null 2>&1 & )
      bench "$out/rep$r" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
      sudo pkill -INT -x pcm-numa; sleep 2
      cpu_summary "$out/rep$r/cpu_util.txt" | tee -a "$RUN_LOG"
    done
    docker exec "$CN" bash -c "tail -60 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  else
    log "cell $name boot failed: $(tail -3 "$out/error_lines.log")"
  fi
  stop_server; unset BOOT_ENV
}
log "== IDE_070 B-series $TS cells=$CELLS reps=$REPS =="
stop_server
for c in ${CELLS//,/ }; do
  case $c in
    b1_interleave) run_cell b1_interleave "numactl --interleave=all env" 96 2 ;;
    b2_membind0)   run_cell b2_membind0   "numactl --membind=0 env"     96 2 ;;
    b3_pool1)      run_cell b3_pool1      ""                             96 1 ;;
    b4_local)      run_cell b4_local      "numactl --cpunodebind=0 --membind=0 env" 56 1 ;;
    *) log "unknown cell $c" ;;
  esac
done
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
for d in $BASE/*/; do n=$(basename $d); for r in $d/rep*/; do [ -f $r/bench_summary.txt ] && log "$n $(basename $r): $(grep -E 'Output token|Median TPOT|P95 TPOT' $r/bench_summary.txt | tr '\n' ' ')"; done; done
log "== B-series done =="
echo "$BASE"
