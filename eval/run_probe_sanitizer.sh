#!/usr/bin/env bash
# [Phase E16] compute-sanitizer memcheck — 60s sample.
# enforce_eager=true 필수 (cudagraph 충돌). 큰 overhead → 짧은 영역.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="E16_sanitizer_memcheck"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# v1.5 env (chain firing 95%+)
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
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO VLLM_NEO_DISABLE_D5
unset VLLM_NEO_D12_TOKEN_MARGIN VLLM_NEO_SWAP_COOLDOWN
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE VLLM_DEBUG_CDEC_PATH

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[E16-sanitizer] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

# compute-sanitizer memcheck. enforce_eager=true 필수.
# --target-processes all → child process 모두 trace
# --launch-timeout 0 → no timeout
# --error-exitcode 0 → error 시 종료 안 함 (run 끝까지)
taskset -c 0-111 /usr/local/cuda/bin/compute-sanitizer \
    --tool memcheck \
    --target-processes all \
    --launch-timeout 0 \
    --error-exitcode 0 \
    --log-file "${OUT_DIR}/sanitizer_memcheck.log" \
    "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 100 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager true \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[E16-sanitizer] launcher PID=${LAUNCHER_PID}"

# Compute-sanitizer 영역은 매우 느림. 360초 sample.
sleep 360

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null
pgrep -f "compute-sanitizer\|run_neo_baseline" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[E16-sanitizer] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== sanitizer error summary ====="
grep -cE "ERROR:|out of bounds|Invalid|race|uninitialized" "${OUT_DIR}/sanitizer_memcheck.log" 2>/dev/null
echo ""
echo "===== sanitizer 마지막 30줄 ====="
tail -30 "${OUT_DIR}/sanitizer_memcheck.log" 2>/dev/null
echo ""
echo "===== crash ====="
grep -ciE "died unexpectedly|EngineDeadError|fatal" "${LOG_FILE}.stdout" 2>/dev/null
echo "[E16-sanitizer] done"
