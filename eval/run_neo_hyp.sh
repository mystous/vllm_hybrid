#!/usr/bin/env bash
# NEO 가설 측정 — 100p, configurable env via positional/named overrides.
# Usage: run_neo_hyp.sh <tag> [KEY=VAL ...]
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tag> [KEY=VAL ...]"
    exit 1
fi

TAG_RAW="$1"
shift
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="hyp_${TAG_RAW}"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# 알고리즘 영역 임의 env 영역 전부 unset (run_neo_clean 와 동일)
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

# Default 영역 (run_neo_clean 와 동일 baseline)
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_CPU_RESIDENT_REQS=128
export VLLM_NEO_PROFILE=1

export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

# Default gpu_memory_utilization (overridable via positional)
GMU=0.85

# Positional overrides — KEY=VAL 형식
for E in "$@"; do
    if [[ "$E" == GMU=* ]]; then
        GMU="${E#GMU=}"
    elif [[ "$E" == *=* ]]; then
        export "$E"
    fi
done

echo "[${TAG}] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[${TAG}] override args: $@"
echo "[${TAG}] active VLLM_NEO_* env:"
env | grep -E "^VLLM_NEO_" | sort
echo "[${TAG}] gpu_memory_utilization=${GMU}"
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization ${GMU} \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 100 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[${TAG}] launcher PID=${LAUNCHER_PID}"

# Polling 방식 — `wait` builtin 가 nested shell 환경에서 즉시 127 반환
# 회피. kill -0 로 launcher PID alive 추적.
START_TS=$(date +%s)
MAX_WAIT=3600
while kill -0 ${LAUNCHER_PID} 2>/dev/null; do
    sleep 15
    NOW_TS=$(date +%s)
    if [ $((NOW_TS - START_TS)) -gt ${MAX_WAIT} ]; then
        echo "[${TAG}] timeout — killing launcher"
        kill -9 ${LAUNCHER_PID} 2>/dev/null
        break
    fi
done
EXIT_CODE=0
ELAPSED=$(($(date +%s) - START_TS))
echo "[${TAG}] launcher exited after ${ELAPSED}s"

pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[${TAG}] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== output_tps ====="
cat "${OUT_DIR}/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('output_tps=', d.get('output_tps'), 'wall_s=', d.get('generate_wall_s'))" 2>/dev/null
echo ""
echo "===== swap counts (TP0) ====="
echo "async swap_out: $(grep 'PROFILE SWAP_OUT async' ${LOG_FILE}.stdout 2>/dev/null | grep -c Worker_TP0)"
echo "sync  swap_out: $(grep 'PROFILE SWAP_OUT[^_]' ${LOG_FILE}.stdout 2>/dev/null | grep -c Worker_TP0)"
echo ""
echo "===== mirror size 최종 ====="
grep "mirror_set_size" "${LOG_FILE}.stdout" 2>/dev/null | tail -3 | grep -oE "mirror_set_size=[0-9]+"
echo ""
echo "===== deadlock escape ====="
grep "NEO deadlock escape" "${LOG_FILE}.stdout" 2>/dev/null | head -2 || echo "미발화"
echo ""
echo "===== crash ====="
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' ${LOG_FILE}.stdout 2>/dev/null)"
echo "[${TAG}] analysis done"
