#!/usr/bin/env bash
# [D-cdec-trace] cdec dispatch path 진입 조건 fire fact 추적.
# 5분 short — engine init 4분 + chain firing 영역 1분.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="cdec_trace_short"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === v1.5.1 env ===
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
unset VLLM_DEBUG_FAULTHANDLER

# === D-cdec-trace 활성 ===
export VLLM_DEBUG_CDEC_PATH=1
unset VLLM_NEO_PROFILE  # cdec trace 만

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[cdec_trace] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[cdec_trace] launcher PID=${LAUNCHER_PID}"

sleep 300

# Cleanup (direct PID kill)
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 5
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null

echo "[cdec_trace] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -5
echo ""
echo "===== D-cdec-trace fire count breakdown ====="
grep "D-cdec-trace" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== sub_batches output attach ====="
grep -c "neo_sub_batches\|neo_sub_batch_cdec_seq_slices\|swap_out attach\|G/H" "${LOG_FILE}.stdout" 2>/dev/null
echo "lines"
echo ""
echo "===== FORK STAT ====="
grep "FORK STAT" "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo ""
echo "===== crash ====="
grep -ciE "died unexpectedly|EngineDeadError|fatal" "${LOG_FILE}.stdout" 2>/dev/null
echo "[cdec_trace] analysis done"
