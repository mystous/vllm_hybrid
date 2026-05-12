#!/usr/bin/env bash
# [Phase E18] torch.profiler chrome trace — 5min run with E18 env-gated profile.
# active 20 step capture (wait 200 + warmup 5 + active 20 = 225 step).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="E18_torch_profiler"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# v1.5 env
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
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE VLLM_DEBUG_CDEC_PATH

# === E18 torch.profiler 활성 ===
export VLLM_DEBUG_TORCH_PROFILER=1
export VLLM_E18_TRACE_DIR="${OUT_DIR}/traces"
mkdir -p "${VLLM_E18_TRACE_DIR}"

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[E18-torch_profiler] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 100 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[E18-torch_profiler] launcher PID=${LAUNCHER_PID}"

sleep 360

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null

echo "[E18-torch_profiler] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== generated traces ====="
ls -la "${VLLM_E18_TRACE_DIR}/" 2>/dev/null
echo ""
echo "===== trace sizes ====="
du -h "${VLLM_E18_TRACE_DIR}/"*.json 2>/dev/null | head -10
echo "[E18-torch_profiler] done"
