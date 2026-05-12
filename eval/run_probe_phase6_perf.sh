#!/usr/bin/env bash
# [Phase 6 성능 분해] vanilla 4690 대비 15% gap root 추적.
# v1.5.1 + try105 env (chain firing 95.6% 영역).
# 외부 도구 동시 측정 — py-spy / nvidia-smi 200ms 시계열 / /proc / top -H.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="phase6_perf_breakdown"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === try105 env (chain firing 95.6% 영역) ===
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_MIRROR_MIN_BUFFER=8
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
export VLLM_NEO_OPTION_C_FULL_MIRROR=1
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_FUSED_RMSNORM
unset VLLM_NEO_DISABLE_SWAP_IN VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN VLLM_NEO_SWAP_COOLDOWN
unset VLLM_NEO_PROFILE
unset VLLM_DEBUG_FAULTHANDLER  # production 영역 (analysis 도구 off)

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[phase6] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[phase6] launcher PID=${LAUNCHER_PID}"

# === Phase 6.1: nvidia-smi 시계열 (200ms — fine-grain) ===
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.sm,throttle_reasons.active \
    --format=csv -lms 200 > "${OUT_DIR}/gpu_timeseries_200ms.csv" 2>&1 &
NVSMI_PID=$!

# Engine init 4분 wait
sleep 240

# === Phase 6.2: py-spy 전체 8 worker concurrent (60s native + speedscope) ===
WORKER_PIDS=$(pgrep -f "VLLM::Worker_TP[0-7]" 2>/dev/null)
echo "[phase6] worker PIDs: $WORKER_PIDS"
SPY_PIDS=""
for PID in $WORKER_PIDS; do
    "$PYSPY" record -p "$PID" -d 60 -n -t -i \
        -o "${OUT_DIR}/flame_pid${PID}.svg" -f flamegraph \
        > "${OUT_DIR}/spy_${PID}.log" 2>&1 &
    SPY_PIDS="$SPY_PIDS $!"
    "$PYSPY" record -p "$PID" -d 60 -n -t -i \
        -o "${OUT_DIR}/raw_pid${PID}.txt" -f raw \
        > "${OUT_DIR}/spy_raw_${PID}.log" 2>&1 &
    SPY_PIDS="$SPY_PIDS $!"
done

# EngineCore py-spy (Python overhead 영역)
ENGINE_PID=$(pgrep -f "VLLM::EngineCore" 2>/dev/null | head -1)
if [ -n "$ENGINE_PID" ]; then
    "$PYSPY" record -p "$ENGINE_PID" -d 60 -n -t -i \
        -o "${OUT_DIR}/flame_engine_${ENGINE_PID}.svg" -f flamegraph \
        > "${OUT_DIR}/spy_engine_${ENGINE_PID}.log" 2>&1 &
    SPY_PIDS="$SPY_PIDS $!"
fi

# === Phase 6.3: top -H -b — per-thread CPU 활용 (60s sample) ===
PRIMARY_WORKER=$(echo "$WORKER_PIDS" | head -1)
if [ -n "$PRIMARY_WORKER" ]; then
    top -H -b -d 1 -n 60 -p "$PRIMARY_WORKER" \
        > "${OUT_DIR}/top_H_worker_${PRIMARY_WORKER}.txt" 2>&1 &
fi

# === Phase 6.4: /proc snapshot 시계열 (2s 간격, 60s) ===
{
    for i in $(seq 1 30); do
        TS_INNER=$(TZ=Asia/Seoul date +%H%M%S)
        for PID in $WORKER_PIDS $ENGINE_PID; do
            cat "/proc/$PID/stat" 2>/dev/null > "${OUT_DIR}/proc_stat_${PID}_${TS_INNER}.txt"
            cat "/proc/$PID/status" 2>/dev/null > "${OUT_DIR}/proc_status_${PID}_${TS_INNER}.txt"
        done
        sleep 2
    done
} > "${OUT_DIR}/proc_snap.log" 2>&1 &

# 60s py-spy 대기
wait $SPY_PIDS 2>/dev/null
echo "[phase6] py-spy done $(TZ=Asia/Seoul date -Iseconds)"

# 추가 60s 진행 (안정 영역 추가 sample)
sleep 60

# === cleanup ===
kill $NVSMI_PID 2>/dev/null
wait 2>/dev/null
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[phase6] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== throughput stats ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" \
    | awk '{sum+=$1; if($1>max||NR==1)max=$1; if($1<min||NR==1)min=$1; n++} END {if (n>0) printf "n=%d avg=%.1f min=%.1f max=%.1f\n", n, sum/n, min, max}'
echo ""
echo "===== GPU utilization timeseries summary (200ms) ====="
awk -F, 'NR>1 {n++; gsub(/ /,"",$5); if($5+0>0) {sum+=$5; nact++}} END {if(n>0) printf "samples=%d avg_gpu_util=%.1f%% nonzero=%d\n", n, sum/nact, nact}' "${OUT_DIR}/gpu_timeseries_200ms.csv" 2>/dev/null
echo ""
echo "===== /proc snapshot count ====="
ls "${OUT_DIR}"/proc_stat_*.txt 2>/dev/null | wc -l
echo ""
echo "===== py-spy flamegraph 생성 ====="
ls -la "${OUT_DIR}"/flame_*.svg 2>/dev/null | head -10
echo ""
echo "===== chain firing fact (sanity) ====="
echo "BUF ALLOC max: $(grep 'BUF ALLOC' "${LOG_FILE}.stdout" 2>/dev/null | grep -oE 'count=[0-9]+' | sed 's/count=//' | sort -n | tail -1)"
echo "FORK STAT: $(grep 'FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1)"
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "[phase6] analysis done"
