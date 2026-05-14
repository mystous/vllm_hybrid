#!/usr/bin/env bash
# [Step 2 — 30분 full run] NameError fix 후 try105 의 21분 crash 영역 도달 검증.
# faulthandler ON (분석 보강) + 동일 v1.4 env (chain firing 98.7%).
# Step 1 v2 의 7분 sample 에서 crash 0 + throughput 760 tps 확보 — 30분 sustain 검증.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="step2_30min_nameerror_fix"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

# === Phase 1: faulthandler ON (분석 보강) + ulimit core dump ===
ulimit -c unlimited
export VLLM_DEBUG_FAULTHANDLER=1

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === try105 env (chain firing 98.7%) ===
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
unset VLLM_NEO_OPTION_O2
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

echo "[step2] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[step2] VLLM_DEBUG_FAULTHANDLER=${VLLM_DEBUG_FAULTHANDLER}, ulimit -c = $(ulimit -c)"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[step2] launcher PID=${LAUNCHER_PID}"

# === 30 분 full run ===
sleep 1800

# Cleanup — pgrep -f pattern 이 일부 cycle fail. 직접 PID kill (ps -o pid,comm)
# 으로 보강. cdec_trace / phase6_full / cdec_eager 의 동일 패턴.
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null

echo "[step2] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== throughput (last 10) ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -10
echo ""
echo "===== throughput 평균 (마지막 안정 영역) ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | tail -50 | grep -oE "[0-9]+\.[0-9]+" \
    | awk '{sum+=$1; n++} END {if (n>0) printf "avg=%.1f tps n=%d\n", sum/n, n}'
echo ""
echo "===== chain firing + swap migration ====="
echo "BUF ALLOC max: $(grep "BUF ALLOC" "${LOG_FILE}.stdout" 2>/dev/null | grep -oE "count=[0-9]+" | sed 's/count=//' | sort -n | tail -1)"
echo "swap-in done: $(grep -c "swap-in.*done" "${LOG_FILE}.stdout" 2>/dev/null)"
echo "swap-out done: $(grep -c "SWAP_OUT CALL\|swap-out.*done" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== Processed prompts ====="
grep -oE 'Processed prompts:[^,]*' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== crash check ====="
echo "crash count: $(grep -ciE "died unexpectedly|EngineDeadError|Fatal Python error" "${LOG_FILE}.stdout" 2>/dev/null)"
grep -iE "died unexpectedly|EngineDeadError|Fatal Python error" "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo ""
echo "===== NameError check (Step 3 fix 효과) ====="
echo "NameError count: $(grep -c "_os_th.*not defined" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "[step2] analysis done"
