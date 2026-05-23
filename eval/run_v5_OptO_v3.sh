#!/usr/bin/env bash
# [Plan v5 — Option O v3] try87 회차.
# Goal: try86 (432 tps) 의 swap_in 빈도 1.1/sec 더 줄여 throughput 추가 회복.
# 코드 변경 X — env 만 강화:
#   MAX_SWAP_IN_PER_STEP: 4 → 1  (매 step 1 reqs 만 swap_in)
#   SWAP_COOLDOWN: 20 → 50       (D15+D16 후 D4 silent 영역 늘림)
# 30분 ETA 초과 시 mid-stop.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try87_v5_OptO_v3"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

# Option O v3 — 추가 강화
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=1    # v3: 4 → 1
export VLLM_NEO_SWAP_COOLDOWN=50          # v3: 20 → 50
export VLLM_NEO_FORCE_SWAP_IN=0           # v2 동일
export VLLM_NEO_MIRROR_MIN_BUFFER=4       # v2 동일

# 기존 Option I/K/C/L/M2 + O2 v2 (코드) 유지
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
unset VLLM_NEO_OPTION_A

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

echo "[v5-OptO-v3] starting (MAX_SWAP_IN=1, COOLDOWN=50) → ${LOG_FILE}"
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
    > "${LOG_FILE}.stdout" 2>&1
LAUNCHER_RC=$?
echo "[v5-OptO-v3] launcher exit=${LAUNCHER_RC}"
echo "[v5-OptO-v3] DONE $(date -Iseconds)"
