#!/usr/bin/env bash
# Vanilla clean run — NEO 영역 비활성 (--enable-neo-asymmetric flag 영역
# 미설정). 같은 workload 영역으로 NEO 영역 vs vanilla 영역 비교 baseline.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="vanilla_clean"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# VLLM_NEO_* 영역 전부 unset — vanilla 영역 보장
unset $(env | grep -oE "^VLLM_NEO_[A-Z_0-9]+" | sort -u) 2>/dev/null
unset VLLM_NEO_PROFILE
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH ENABLE_NEO_INV

# 환경 영역 — hardware tuning. NEO 와 동일 영역 유지 (비교 공정성).
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[vanilla_clean] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[vanilla_clean] active VLLM_NEO_* env (should be none):"
env | grep -E "^VLLM_NEO_" | sort
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[vanilla_clean] launcher PID=${LAUNCHER_PID}"

echo "[vanilla_clean] waiting for completion (timeout=60min)..."
timeout 3600 wait ${LAUNCHER_PID} 2>/dev/null
EXIT_CODE=$?
echo "[vanilla_clean] launcher exit_code=${EXIT_CODE}"

pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[vanilla_clean] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== throughput tail ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -10
echo ""
echo "===== final result.json ====="
cat "${OUT_DIR}/result.json" 2>/dev/null | head -50
echo ""
echo "===== crash check ====="
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[vanilla_clean] analysis done"
