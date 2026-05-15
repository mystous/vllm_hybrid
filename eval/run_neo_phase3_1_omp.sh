#!/usr/bin/env bash
# Phase 3.1 — Persistent OMP team (코드 변경 적용) (NUMA local 검증, env 변경 없음 = v1.6 best + Phase 3.1 OMP persistent code change)
# Phase 3.2 (KMP_BLOCKTIME=0) 비교의 baseline.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="phase3_1_omp"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === v1.6 best + Phase 3.1 OMP persistent code change — Option C/L/M2 OFF (minimal env) ===
unset VLLM_NEO_OPTION_A VLLM_NEO_OPTION_C VLLM_NEO_OPTION_K
unset VLLM_NEO_OPTION_L VLLM_NEO_OPTION_M2 VLLM_NEO_OPTION_O2
unset VLLM_NEO_OPTION_C_FULL_MIRROR
unset VLLM_NEO_SWAP_OUT_RATIO VLLM_NEO_PREDICTIVE_THRESHOLD
unset VLLM_NEO_LOAD_AWARE_MIN_RUNNING
unset VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP
unset VLLM_NEO_MAX_SWAP_IN_PER_STEP
unset VLLM_NEO_MIRROR_MIN_BUFFER VLLM_NEO_MIN_RUNNING_DECODE
unset VLLM_NEO_SWAP_COOLDOWN
unset VLLM_NEO_FORCE_SWAP_IN VLLM_NEO_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_NEOSCHED_STEP23 VLLM_NEO_DRIVE_6STEP VLLM_NEO_6STEP_DRY_RUN
unset VLLM_NEO_DECIDE_MODE_BALANCE
unset VLLM_NEO_HEURISTIC_LINR_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_PREF_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_GDEC_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_CDEC_PER_TOKEN_PAIR_MS
unset VLLM_NEO_HEURISTIC_LNCH_MS
unset VLLM_NEO_SWAP_IN_ORDER
unset VLLM_NEO_ASYNC_CDEC VLLM_NEO_CDEC_PIPELINE_DEPTH
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH
unset ENABLE_NEO_INV

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_CPU_RESIDENT_REQS=128
export VLLM_NEO_ASYNC_SWAP_BUFFERS=3
export VLLM_NEO_PROFILE=1

# === hardware tuning — v1.6 best 동일 (KMP_BLOCKTIME default = 200ms) ===
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
unset KMP_BLOCKTIME
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

echo "[phase3_1_omp] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[phase3_1_omp] 400p × 8192, v1.6 best env, (코드: omp_set_dynamic(0) + max_active_levels(1) in core.h ispc_attention_tasks)"
env | grep -E "^VLLM_NEO_|^KMP_|^OMP_" | sort
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.92 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 400 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[phase3_1_omp] launcher PID=${LAUNCHER_PID}"

# Wait 4 min then capture NUMA placement (Phase 3.4 검증)
NUMA_LOG="${OUT_DIR}/numa_placement.txt"
START_TS=$(date +%s)
SNAPSHOT_DONE=0
MAX_WAIT=5400  # 90 min
while kill -0 ${LAUNCHER_PID} 2>/dev/null; do
    sleep 30
    NOW_TS=$(date +%s)
    ELAPSED_NOW=$((NOW_TS - START_TS))
    # Capture NUMA placement once after 4 min (warmup 후)
    if [ ${SNAPSHOT_DONE} -eq 0 ] && [ ${ELAPSED_NOW} -gt 240 ]; then
        echo "=== NUMA placement snapshot at ${ELAPSED_NOW}s ===" > "${NUMA_LOG}"
        for i in 0 1 2 3 4 5 6 7; do
            pid=$(ps -eo pid,cmd | grep "VLLM::Worker_TP${i}\$" | grep -v grep | awk '{print $1}' | head -1)
            if [ -n "$pid" ]; then
                {
                    echo ""
                    echo "--- TP${i} pid=${pid} ---"
                    numastat -p "$pid" 2>/dev/null | head -15
                } >> "${NUMA_LOG}"
            fi
        done
        echo "[phase3_1_omp] NUMA snapshot → ${NUMA_LOG}"
        SNAPSHOT_DONE=1
    fi
    if [ ${ELAPSED_NOW} -gt ${MAX_WAIT} ]; then
        echo "[phase3_1_omp] timeout — killing launcher"
        kill -9 ${LAUNCHER_PID} 2>/dev/null
        break
    fi
done
ELAPSED=$(($(date +%s) - START_TS))
echo "[phase3_1_omp] launcher exited after ${ELAPSED}s"

pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[phase3_1_omp] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== output_tps ====="
cat "${OUT_DIR}/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"output_tps={d['output_tps']:.1f} wall={d['generate_wall_s']:.0f}s\")" 2>/dev/null
echo ""
echo "===== 22 strict items ====="
SO=$(grep -c '\[NEO SWAP_OUT CALL\]' ${LOG_FILE}.stdout 2>/dev/null)
SI=$(grep -c 'swap-in: req' ${LOG_FILE}.stdout 2>/dev/null)
echo "  swap_out=${SO} swap_in=${SI}"
FORK=$(grep 'NEO FORK STAT' ${LOG_FILE}.stdout 2>/dev/null | grep Worker_TP0 | tail -1)
echo "  ${FORK}"
PROF=$(grep 'PROFILE PER-LAYER' ${LOG_FILE}.stdout 2>/dev/null | grep Worker_TP0 | tail -1)
echo "  ${PROF}"
MIS=$(grep -c 'shape mismatch' ${LOG_FILE}.stdout 2>/dev/null)
echo "  shape_mismatch=${MIS}"
echo ""
echo "===== NUMA fact ====="
[ -f "${NUMA_LOG}" ] && head -50 "${NUMA_LOG}"
