#!/usr/bin/env bash
# [try97] pacpu kernel 안의 *각 tid 의 first_fire* log 측정.
# 14 thread 모두 log 나오면 OpenMP 정상, 일부만 나오면 OpenMP runtime 결함.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try97_v5_tid_fire"
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
export VLLM_NEO_FORCE_SWAP_IN=0
export VLLM_NEO_SWAP_COOLDOWN=20
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
unset VLLM_NEO_PROFILE

# OMP env 기본 (try96 처럼) — affinity 정책 비활성, places 무영향
export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[try97] starting (각 tid fire log) → ${LOG_FILE}"
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
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
echo "[try97] launcher PID=${LAUNCHER_PID}"

# 7 분 — init 4 + 측정 3
sleep 420

# Force kill
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[try97] DONE $(date -Iseconds)"

# Analysis
echo ""
echo "===== [OMP TID] first_fire log (각 tid 별) ====="
grep '\[OMP TID\]' "${LOG_FILE}.stdout" 2>/dev/null | head -30
echo ""
echo "===== TID 별 fire count ====="
grep '\[OMP TID\]' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE 'tid=[0-9]+' | sort | uniq -c | sort -rn
echo ""
echo "===== [OMP DBG] (ws/num_threads) ====="
grep '\[OMP DBG\]' "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
