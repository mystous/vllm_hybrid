#!/usr/bin/env bash
# Vanilla 단축 측정 — 50 prompts, NEO 비활성
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="vanilla_short"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

unset $(env | grep -oE "^VLLM_NEO_[A-Z_0-9]+" | sort -u) 2>/dev/null
unset VLLM_NEO_PROFILE
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH ENABLE_NEO_INV

export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[vanilla_short] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[vanilla_short] active VLLM_NEO_* env (should be none):"
env | grep -E "^VLLM_NEO_" | sort
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 50 \
    --target-input-len 8192 --max-tokens 8192 \
    --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[vanilla_short] launcher PID=${LAUNCHER_PID}"

echo "[vanilla_short] waiting for completion (timeout=60min)..."
timeout 3600 wait ${LAUNCHER_PID} 2>/dev/null
EXIT_CODE=$?
echo "[vanilla_short] launcher exit_code=${EXIT_CODE}"

pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[vanilla_short] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== output_tps ====="
cat "${OUT_DIR}/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('output_tps=', d.get('output_tps'), 'generate_wall_s=', d.get('generate_wall_s'))" 2>/dev/null
echo ""
echo "===== throughput tail ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -5
echo ""
echo "===== crash check ====="
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[vanilla_short] analysis done"
