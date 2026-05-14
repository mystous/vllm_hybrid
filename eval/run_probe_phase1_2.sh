#!/usr/bin/env bash
# [Phase 1+2] prove 기반 분석 — silent SEGV root 추적.
# - Phase 1: faulthandler (worker startup 영역에 코드 추가됨) + ulimit -c unlimited + systemd-coredump
# - Phase 2: py-spy × 3 worker concurrent + nvidia-smi timeseries + /proc snapshot 시계열
# 5분 reproducer (try105 env 그대로 — chain firing 98.7%).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="probe_phase1_2"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

# === Phase 1: core dump + faulthandler ===
ulimit -c unlimited
export PYTHONFAULTHANDLER=1

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === try105 env (chain firing 98.7%) — 재현성 보장 ===
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
unset VLLM_NEO_OPTION_O2
unset VLLM_NEO_OPTION_A
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_SWAP_COOLDOWN
unset VLLM_NEO_PROFILE

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[probe] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[probe] ulimit -c = $(ulimit -c)"

# Engine launch (background)
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[probe] launcher PID=${LAUNCHER_PID}"

# === Phase 2.2: nvidia-smi GPU state 시계열 (500ms) — 즉시 시작 ===
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.sm,throttle_reasons.active \
    --format=csv -lms 500 > "${OUT_DIR}/gpu_timeseries.csv" 2>&1 &
NVSMI_PID=$!

# === Engine init wait (4분, ramp-up 영역 진입) ===
sleep 240

# === Phase 2.1: py-spy × 3 worker (engine init 후, chain firing 영역) ===
WORKER_PIDS=$(pgrep -f "VLLM::Worker_TP[0-7]" 2>/dev/null | head -3)
echo "[probe] worker PIDs for py-spy: $WORKER_PIDS"
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

# === Phase 2.3: /proc snapshot 시계열 (engine init 후 5초 간격) ===
{
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
        TIMESTAMP=$(TZ=Asia/Seoul date +%H%M%S)
        for PID in $(pgrep -f "VLLM::Worker_TP[0-7]" 2>/dev/null); do
            cat "/proc/$PID/status" 2>/dev/null > "${OUT_DIR}/proc_status_${PID}_${TIMESTAMP}.txt"
        done
        EngineCorePID=$(pgrep -f "VLLM::EngineCore" 2>/dev/null | head -1)
        if [ -n "$EngineCorePID" ]; then
            cat "/proc/$EngineCorePID/status" 2>/dev/null > "${OUT_DIR}/proc_status_engine_${TIMESTAMP}.txt"
        fi
        sleep 5
    done
} &
PROC_SNAP_PID=$!

# === 60s py-spy 측정 + cleanup ===
wait $SPY_PIDS 2>/dev/null
echo "[probe] py-spy done $(TZ=Asia/Seoul date -Iseconds)"

# 추가 30s wait — silent SEGV 영역 capture (worker 가 죽으면 faulthandler trace 출력)
sleep 30

# nvidia-smi + /proc snapshot stop
kill $NVSMI_PID $PROC_SNAP_PID 2>/dev/null
wait 2>/dev/null

# Engine cleanup
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[probe] $(TZ=Asia/Seoul date -Iseconds) DONE"

# === Phase 1/2 analysis ===
echo ""
echo "===== throughput (last 5) ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -5

echo ""
echo "===== crash trace (faulthandler / died unexpectedly) ====="
grep -iE 'died unexpectedly|EngineDeadError|fatal|Fatal Python error|Current thread.*ThreadID|Segmentation fault' \
    "${LOG_FILE}.stdout" 2>/dev/null | head -20

echo ""
echo "===== faulthandler stack (Python+C stack trace 영역) ====="
awk '/Fatal Python error|Current thread.*ThreadID/,/^$/' "${LOG_FILE}.stdout" 2>/dev/null | head -50

echo ""
echo "===== core dump (systemd-coredump 영역) ====="
ls /var/lib/systemd/coredump/ 2>/dev/null | tail -5

echo ""
echo "===== py-spy flamegraph 결과 ====="
ls -la "${OUT_DIR}"/flame_pid*.svg 2>/dev/null

echo ""
echo "===== GPU state (crash 직전 영역) ====="
tail -20 "${OUT_DIR}/gpu_timeseries.csv" 2>/dev/null

echo ""
echo "===== /proc snapshot count ====="
ls "${OUT_DIR}"/proc_status_*.txt 2>/dev/null | wc -l
echo "snapshot files"

echo ""
echo "[probe] $(TZ=Asia/Seoul date -Iseconds) analysis done"
