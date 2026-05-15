#!/usr/bin/env bash
# ASYNC_SWAP sweep — TSK_019 v1.7 Phase E follow-up (OQ04).
#
# 목적: VLLM_NEO_ASYNC_SWAP=1 (baseline) vs 0 (sync only) wall 차이 측정.
#       Phase E `E_open_questions.md` OQ 영역에 reflect.
#
# 비교를 위해 동일 commit + 동일 workload + 동일 hardware 에서 양쪽 모두
# 신규 측정 (기존 500p baseline 는 시점·체크포인트 다름).
#
# workload: 200p × 8192 in/out (full 500p 의 40%, ~ 10-13min/run 예상).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

# 인자1: 0 = sync only, 1 = async
# 인자2: ASYNC_SWAP_BUFFERS (default 3, range 1..8)
MODE="${1:?usage: $0 <0|1> [buffers=3]}"
BUFFERS="${2:-3}"
case "$MODE" in
    0) TAG="async0_sync_b${BUFFERS}"; ASYNC_VAL=0 ;;
    1) TAG="async1_b${BUFFERS}"; ASYNC_VAL=1 ;;
    *) echo "MODE must be 0 or 1"; exit 1 ;;
esac

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === 실험적 NEO env 모두 unset (run_neo_standard.sh 와 동일) ===
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

# === 표준 NEO env ===
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_CPU_RESIDENT_REQS=128
export VLLM_NEO_ASYNC_SWAP_BUFFERS=${BUFFERS}
export VLLM_NEO_PROFILE=1

# === sweep 영역 — 본 측정의 유일한 차이 ===
export VLLM_NEO_ASYNC_SWAP=${ASYNC_VAL}

# === hardware tuning (NUMA + OMP) ===
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

echo "[async_sweep MODE=${MODE}] $(TZ=Asia/Seoul date -Iseconds) → ${OUT_DIR}"
echo "[async_sweep MODE=${MODE}] VLLM_NEO_ASYNC_SWAP=${ASYNC_VAL}"
echo "[async_sweep MODE=${MODE}] active VLLM_NEO_* env:"
env | grep -E "^VLLM_NEO_" | sort
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.92 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 200 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[async_sweep] launcher PID=${LAUNCHER_PID}"

START_TS=$(date +%s)
MAX_WAIT=3600  # 60min safety
while kill -0 ${LAUNCHER_PID} 2>/dev/null; do
    sleep 15
    NOW_TS=$(date +%s)
    if [ $((NOW_TS - START_TS)) -gt ${MAX_WAIT} ]; then
        echo "[async_sweep] timeout — killing launcher"
        kill -9 ${LAUNCHER_PID} 2>/dev/null
        break
    fi
done
ELAPSED=$(($(date +%s) - START_TS))
echo "[async_sweep MODE=${MODE}] launcher exited after ${ELAPSED}s"

pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[async_sweep MODE=${MODE}] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== output_tps ====="
cat "${OUT_DIR}/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('output_tps=', d.get('output_tps'), 'wall_s=', d.get('generate_wall_s'))" 2>/dev/null
echo ""
echo "===== swap counts (TP0) ====="
# Note: grep -c 'sync' alone matches "Asynchronous" too. Use literal '(async)' / '(sync)'.
echo "  async: $(grep -c '\[NEO SWAP_OUT CALL\].*(async)' ${LOG_FILE}.stdout 2>/dev/null)"
echo "  sync:  $(grep -c '\[NEO SWAP_OUT CALL\].*(sync)' ${LOG_FILE}.stdout 2>/dev/null)"
echo ""
echo "===== crash check ====="
echo "  crash: $(grep -ciE 'died unexpectedly|EngineDeadError' ${LOG_FILE}.stdout 2>/dev/null)"
echo "===== summary ====="
echo "  OUT_DIR=${OUT_DIR}"
