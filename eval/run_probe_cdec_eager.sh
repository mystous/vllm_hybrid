#!/usr/bin/env bash
# [cdec-eager] enforce_eager=true 로 cudagraph 비활성 — Python branch 항상 fire.
# cdec_future submit 의 진정한 fire 영역 확인.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"
TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="cdec_eager_short"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"
ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
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
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A VLLM_DEBUG_FAULTHANDLER
unset VLLM_NEO_PROFILE
export VLLM_DEBUG_CDEC_PATH=1
export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[cdec_eager] $(TZ=Asia/Seoul date -Iseconds) starting"

# === enforce_eager=true (cudagraph 비활성) ===
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager true \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[cdec_eager] launcher PID=${LAUNCHER_PID}"

sleep 300

ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 5
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null

echo "[cdec_eager] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== D-cdec-trace (enforce_eager=true) ====="
grep "D-cdec-trace" "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo ""
echo "===== NEO CDEC CALL fire ====="
grep -c "NEO CDEC CALL" "${LOG_FILE}.stdout" 2>/dev/null
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -5
echo "[cdec_eager] done"
