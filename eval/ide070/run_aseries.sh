#!/usr/bin/env bash
# IDE_070 / TSK_055 — A 시리즈 (코드 무변경). 기준선 = hot96 def4 KV40k TP4, C64, 각 셀 3회 반복.
#   A1 turbo ON   : intel_pstate/no_turbo 0 → 측정 후 반드시 1 로 원복 (k8s 워커 공유 설정)
#   A4 worker     : cpuinfer 80 / 112 (96 은 기준선)
#   A6 GPU 토폴로지: CUDA_VISIBLE_DEVICES=0,1,4,5 (양 소켓 분산) — 기준선 0,1,2,3 과 대조
# 사용: run_aseries.sh <cells: a0,a1,a4_80,a4_112,a6>
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
# a1 (turbo ON) 은 이 노드에서 BIOS 잠금으로 no_turbo 쓰기가 root 로도 거부됨 (IDE_030 8-30 기록, 9-15 재확인) → 기본 목록에서 제외
CELLS=${1:-a0,a4_80,a4_112,a6}; REPS=${REPS:-3}; C=64; N=$((C*4))
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_aseries
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
base_args() {  # base_args <cpuinfer>
  echo "--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer $1 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic"
}
turbo_set() { echo "$1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null; sleep 1; log "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo) max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"; }
trap 'turbo_set 1' EXIT   # 어떤 경우에도 turbo OFF 로 복귀

run_cell() {  # run_cell <name> <gpus> <cpuinfer> <turbo 0/1>
  local name=$1 gpus=$2 ci=$3 tb=$4
  local out=$BASE/$name
  log "-- cell $name gpus=$gpus cpuinfer=$ci no_turbo=$tb --"
  turbo_set "$tb"
  boot "$out" "$gpus" 2400 $(base_args $ci)
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    bench "$out/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1
    for r in $(seq 1 $REPS); do
      ( sudo turbostat --interval 5 --quiet --show Avg_MHz,Busy%,Bzy_MHz > "$out/rep$r.turbostat.txt" 2>&1 & echo $! > "$out/.turbo" )
      bench "$out/rep$r" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
      sudo pkill -f "turbostat --interval" 2>/dev/null
      cpu_summary "$out/rep$r/cpu_util.txt" | tee -a "$RUN_LOG"
      grep -vE "^Avg|^$" "$out/rep$r.turbostat.txt" | awk 'NF==3{s+=$3;n++} END{if(n) printf "Bzy_MHz avg %.0f (n=%d)\n", s/n, n}' | tee -a "$RUN_LOG"
    done
    docker exec "$CN" bash -c "tail -60 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  fi
  stop_server; turbo_set 1
}
log "== IDE_070 A-series $TS cells=$CELLS reps=$REPS =="
stop_server
for c in ${CELLS//,/ }; do
  case $c in
    a0)      run_cell a0_base_turboOFF   0,1,2,3 96  1 ;;
    a1)      run_cell a1_turboON         0,1,2,3 96  0 ;;
    a4_80)   run_cell a4_cpu80           0,1,2,3 80  1 ;;
    a4_112)  run_cell a4_cpu112          0,1,2,3 112 1 ;;
    a6)      run_cell a6_gpu0145         0,1,4,5 96  1 ;;
    a6t)     run_cell a6_gpu0145_turboON 0,1,4,5 96  0 ;;
    *) log "unknown cell $c" ;;
  esac
done
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
log "== A-series done $(date +%H:%M:%S) =="
echo "$BASE"
