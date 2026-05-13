#!/usr/bin/env bash
# NEO clean run — 알고리즘 영역 임의 env 영역 *전부 제거*. 코드 영역
# 의 자연 default 영역 으로만 NEO 영역 동작. 비교 baseline 영역.
#
# 보존 영역: hardware 영역 (CPU pin / NUMA / OMP) + measurement (PROFILE)
# 제거 영역: VLLM_NEO_OPTION_* / VLLM_NEO_*_RATIO / VLLM_NEO_*_CAP /
#           VLLM_NEO_FORCE_* / VLLM_NEO_*_MIN_* / VLLM_NEO_NEOSCHED_* /
#           VLLM_NEO_DRIVE_* / VLLM_NEO_HEURISTIC_* / 기타 algorithm
#           영역 강제 env
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="neo_clean"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === 알고리즘 영역 임의 env 영역 *전부 unset* ===
# VLLM_NEO_OPTION_* 영역
unset VLLM_NEO_OPTION_A VLLM_NEO_OPTION_C VLLM_NEO_OPTION_K
unset VLLM_NEO_OPTION_L VLLM_NEO_OPTION_M2 VLLM_NEO_OPTION_O2
unset VLLM_NEO_OPTION_C_FULL_MIRROR
# threshold / ratio / cap 영역
unset VLLM_NEO_SWAP_OUT_RATIO VLLM_NEO_PREDICTIVE_THRESHOLD
unset VLLM_NEO_LOAD_AWARE_MIN_RUNNING
unset VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP
unset VLLM_NEO_MAX_SWAP_IN_PER_STEP
unset VLLM_NEO_CPU_RESIDENT_REQS VLLM_NEO_MIRROR_MIN_BUFFER
unset VLLM_NEO_MIRROR_MAX VLLM_NEO_MIN_RUNNING_DECODE
unset VLLM_NEO_SWAP_COOLDOWN
# force / disable 영역
unset VLLM_NEO_FORCE_SWAP_IN VLLM_NEO_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_D12_TOKEN_MARGIN
# scheduler driving 영역
unset VLLM_NEO_NEOSCHED_STEP23 VLLM_NEO_DRIVE_6STEP VLLM_NEO_6STEP_DRY_RUN
# decide_mode 영역
unset VLLM_NEO_DECIDE_MODE_BALANCE
# heuristic 상수 override 영역
unset VLLM_NEO_HEURISTIC_LINR_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_PREF_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_GDEC_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_CDEC_PER_TOKEN_PAIR_MS
unset VLLM_NEO_HEURISTIC_LNCH_MS
# swap order 영역
unset VLLM_NEO_SWAP_IN_ORDER
# async cdec 영역
unset VLLM_NEO_ASYNC_CDEC VLLM_NEO_CDEC_PIPELINE_DEPTH
# debug 영역
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH
unset ENABLE_NEO_INV

# === 보존 영역 — predictor 만 (TablePerfPredictor 영역 ModelProfiler
# 미적재 영역이라 heuristic 영역으로 자연 fallback) ===
export VLLM_NEO_PREDICTOR=heuristic

# CPU KV pool 영역 2× 영역 확장 — default 64 → 128 (ABSOLUTE_CAP 영역).
# per-worker 영역 80 GiB pinned × 8 worker = 640 GiB. NUMA 영역 분배:
# rank 0-3 → NUMA0 (320 GiB / 800 GB free), rank 4-7 → NUMA1 (동일).
# VLLM_NEO_NUMA_BIND=1 영역으로 numa_set_localalloc() 영역 local 영역 강제.
export VLLM_NEO_CPU_RESIDENT_REQS=128

# === 측정 영역 ===
export VLLM_NEO_PROFILE=1

# === 환경 영역 — algorithm 무관, hardware tuning ===
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

echo "[neo_clean] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[neo_clean] active VLLM_NEO_* env:"
env | grep -E "^VLLM_NEO_" | sort
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[neo_clean] launcher PID=${LAUNCHER_PID}"

# 500 prompts 완주 대기 — 자연 종료 영역. timeout 영역 60min (안전).
# python script 영역 result.json 영역 작성 후 정상 종료.
echo "[neo_clean] waiting for completion (timeout=60min)..."
timeout 3600 wait ${LAUNCHER_PID} 2>/dev/null
EXIT_CODE=$?
echo "[neo_clean] launcher exit_code=${EXIT_CODE}"

# residual worker cleanup
pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[neo_clean] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -10
echo ""
echo "===== PROFILE PER-LAYER tail ====="
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== SWAP_OUT/IN counts ====="
echo "swap_out events: $(grep -c "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null)"
echo "swap_in events:  $(grep -c "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== cdec calls ====="
grep "\[NEO CDEC CALL\]" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== crash check ====="
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[neo_clean] analysis done"
