#!/usr/bin/env bash
# NEO async swap 단축 측정 — 50 prompts, 동일 workload 파라미터
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="neo_async_short"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# 알고리즘 env 전부 unset
unset VLLM_NEO_OPTION_A VLLM_NEO_OPTION_C VLLM_NEO_OPTION_K
unset VLLM_NEO_OPTION_L VLLM_NEO_OPTION_M2 VLLM_NEO_OPTION_O2
unset VLLM_NEO_OPTION_C_FULL_MIRROR
unset VLLM_NEO_SWAP_OUT_RATIO VLLM_NEO_PREDICTIVE_THRESHOLD
unset VLLM_NEO_LOAD_AWARE_MIN_RUNNING
unset VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP
unset VLLM_NEO_MAX_SWAP_IN_PER_STEP
unset VLLM_NEO_CPU_RESIDENT_REQS VLLM_NEO_MIRROR_MIN_BUFFER
unset VLLM_NEO_MIRROR_MAX VLLM_NEO_MIN_RUNNING_DECODE
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
export VLLM_NEO_PROFILE=1

export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

echo "[neo_async_short] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[neo_async_short] active VLLM_NEO_* env:"
env | grep -E "^VLLM_NEO_" | sort
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 50 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[neo_async_short] launcher PID=${LAUNCHER_PID}"

echo "[neo_async_short] waiting for completion (timeout=60min)..."
timeout 3600 wait ${LAUNCHER_PID} 2>/dev/null
EXIT_CODE=$?
echo "[neo_async_short] launcher exit_code=${EXIT_CODE}"

pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[neo_async_short] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== output_tps ====="
cat "${OUT_DIR}/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('output_tps=', d.get('output_tps'), 'generate_wall_s=', d.get('generate_wall_s'))" 2>/dev/null
echo ""
echo "===== throughput tail ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -10
echo ""
echo "===== swap counts (TP0) ====="
echo "async swap_out: $(grep -c "PROFILE SWAP_OUT async" "${LOG_FILE}.stdout" 2>/dev/null)"
echo "sync  swap_out: $(grep -c "PROFILE SWAP_OUT\b" "${LOG_FILE}.stdout" 2>/dev/null)"
echo "swap_in:        $(grep -c "PROFILE SWAP_IN\b" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== b1_avg tail ====="
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | grep "Worker_TP0" | tail -3 | grep -oE "b1_avg=[0-9]+ .* b0_avg=[0-9]+"
echo ""
echo "===== crash check ====="
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[neo_async_short] analysis done"
