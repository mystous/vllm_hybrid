#!/usr/bin/env bash
# [Cycle 2.1 분석] Option C v2 (full_mirror=1) 의 부작용 검증.
# try105 (full_mirror=1) 가 21분 silent SEGV → swap migration 5 logs (try84 6590 대비 격감).
# Cycle 2.1: VLLM_NEO_OPTION_C_FULL_MIRROR=0 으로 v1.4 decide_mode 회귀.
# Short 5분 측정 — crash 재현 여부 + swap migration 빈도 회복 검증.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="cycle2_1_optC_off_short"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === v1.4 (try84) env — chain firing 98.7% 발화 ===
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
# === Cycle 2.1 — full_mirror=0 으로 v1.4 decide_mode 회귀 ===
export VLLM_NEO_OPTION_C_FULL_MIRROR=0
unset VLLM_NEO_OPTION_A
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_SWAP_COOLDOWN
unset VLLM_NEO_PROFILE

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[cycle2.1] $(TZ=Asia/Seoul date -Iseconds) starting → ${LOG_FILE}"
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[cycle2.1] launcher PID=${LAUNCHER_PID}"

# === short 5분 measurement (try104 3분 crash 잡힘) ===
sleep 300

pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[cycle2.1] $(TZ=Asia/Seoul date -Iseconds) DONE"

echo ""
echo "===== throughput (last 5) ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo ""
echo "===== swap migration count (BUF ALLOC) ====="
grep "BUF ALLOC" "${LOG_FILE}.stdout" 2>/dev/null | grep -oE "count=[0-9]+" | sed 's/count=//' | sort -n | tail -1
echo "max BUF ALLOC count"
echo ""
echo "===== SWAP_IN dispatch fire ====="
grep -c "SWAP_IN dispatch\|swap_in_attach" "${LOG_FILE}.stdout" 2>/dev/null
echo ""
echo "===== crash ====="
grep -iE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo ""
echo "===== cycle2.1 analysis done ====="
