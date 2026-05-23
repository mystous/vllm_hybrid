#!/usr/bin/env bash
# [Plan v5 — Option O = O1+O2] try85 회차.
# Goal: throughput 회복 (try84 246 tps → ≥1000 tps target).
# Option O 의 4 가지 fix:
#   O1.a: VLLM_NEO_FORCE_SWAP_IN=0 — 95% threshold check 복원
#   O1.b: VLLM_NEO_SWAP_COOLDOWN=20 — D15+D16 후 D4 silent 영역 확보
#   O1.c: VLLM_NEO_MIRROR_MIN_BUFFER=4 — mirror 작게 (cycle 재진입 줄임)
#   O2:   D4 budget check 코드 (max_num_seqs - len(running) 차감)
# 30분 ETA 초과 시 mid-stop (사용자 명시).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try85_v5_OptO"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

# Option O1 — env 조정
export VLLM_NEO_FORCE_SWAP_IN=0           # O1.a: 95% threshold 복원
export VLLM_NEO_SWAP_COOLDOWN=20          # O1.b: cooldown 5 → 20
export VLLM_NEO_MIRROR_MIN_BUFFER=4       # O1.c: MIN_BUFFER 8 → 4

# 기존 Option I/K/C/L/M2 유지
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
unset VLLM_NEO_OPTION_A

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

echo "[v5-OptO] starting (FORCE_SWAP_IN=0, COOLDOWN=20, MIN_BUFFER=4, O2 budget) → ${LOG_FILE}"
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
echo "[v5-OptO] launcher exit=${LAUNCHER_RC}"
echo "[v5-OptO] DONE $(date -Iseconds)"
