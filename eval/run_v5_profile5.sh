#!/usr/bin/env bash
# [Plan v5 — profile5] try89 회차.
# Goal: per-layer GPU forward vs CPU pacpu wait elapsed 측정.
#       overlap 깨진 root + GPU/CPU imbalance 정량.
# 본 회차 = 5분만. 측정 instrumentation 자체가 GPU sync 포함 → 오버헤드 있음.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try92_v5_profile5"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=1
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_FORCE_SWAP_IN=0
export VLLM_NEO_SWAP_COOLDOWN=50
export VLLM_NEO_MIRROR_MIN_BUFFER=4
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
unset VLLM_NEO_OPTION_A
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

# Profile mode 활성 (per-layer + swap)
export VLLM_NEO_PROFILE=1

echo "[v5-profile5] starting (per-layer instrumentation) → ${LOG_FILE}"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --num-prompts 500 \
    --target-input-len 8192 \
    --max-tokens 8192 \
    --enable-neo-asymmetric \
    --async-scheduling \
    --enforce-eager false \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" \
    --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[v5-profile5] launcher PID=${LAUNCHER_PID}"

# 5 min total — init 3분 + 측정 2분
sleep 420
kill -TERM $LAUNCHER_PID 2>/dev/null
sleep 3
kill -9 $LAUNCHER_PID 2>/dev/null
ps -ef | grep "EngineCore\|VLLM::Worker" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

echo "[v5-profile5] DONE $(date -Iseconds)"

# Quick analysis
echo ""
echo "===== PER-LAYER PROFILE 결과 ====="
echo "--- PROFILE PER-LAYER (마지막 5 log) ---"
grep '\[PROFILE PER-LAYER\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo ""
echo "--- throughput ---"
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "[v5-profile5] done"
