#!/usr/bin/env bash
# [Cycle 2.2 short test] Option O2 v2 default 비활성 (env-gated, VLLM_NEO_OPTION_O2 default 0).
# Cycle 2.1 (full_mirror=0) 도 5분 crash → Option C v2 root 아님.
# Cycle 2.2: full_mirror=1 (v1.5 fix 유지) + Option O2 v2 비활성 → swap migration 회복 검증.
# 가설: Option O2 v2 가 swap_in 영구 silent root → swap dispatch loop 손상 → SEGV.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="cycle2_2_optO2_off_short"
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
# === Cycle 2.2 — full_mirror=1 유지, Option O2 v2 default 비활성 ===
export VLLM_NEO_OPTION_C_FULL_MIRROR=1
unset VLLM_NEO_OPTION_O2  # default 0 (비활성)
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

echo "[cycle2.2] $(TZ=Asia/Seoul date -Iseconds) starting → ${LOG_FILE}"
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[cycle2.2] launcher PID=${LAUNCHER_PID}"

# === short 5분 measurement ===
sleep 300

pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[cycle2.2] $(TZ=Asia/Seoul date -Iseconds) DONE"

echo ""
echo "===== throughput (last 5) ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo ""
echo "===== swap migration max count (BUF ALLOC) ====="
grep "BUF ALLOC" "${LOG_FILE}.stdout" 2>/dev/null | grep -oE "count=[0-9]+" | sed 's/count=//' | sort -n | tail -1
echo "(try84 6590 / try105 5 — Cycle 2.2 goal: ≥ 200)"
echo ""
echo "===== SWAP_IN dispatch count ====="
grep -c "SWAP_IN dispatch" "${LOG_FILE}.stdout" 2>/dev/null
echo ""
echo "===== crash ====="
grep -iE 'died unexpectedly|EngineDeadError|fatal' "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo ""
echo "===== cycle2.2 analysis done ====="
