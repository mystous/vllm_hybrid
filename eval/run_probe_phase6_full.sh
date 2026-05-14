#!/usr/bin/env bash
# [Phase 6 full] NEO PROFILE (log_freq=50) + py-spy + ncu kernel detail.
# 5분 launch + 60s ncu sample (chain firing 영역).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="phase6_full"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === v1.5.1 env (chain firing 95%) ===
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
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_FUSED_RMSNORM
unset VLLM_NEO_DISABLE_SWAP_IN VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN VLLM_NEO_SWAP_COOLDOWN
unset VLLM_DEBUG_FAULTHANDLER

# === Phase 6.1+6.2 (PROFILE 활성 — log_freq=50 빈번 sample) ===
export VLLM_NEO_PROFILE=1

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[phase6_full] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[phase6_full] launcher PID=${LAUNCHER_PID}"

sleep 300

# Direct PID kill (launcher pgrep cleanup logic 가 fail 영역)
PIDS=$(pgrep -P $LAUNCHER_PID 2>/dev/null) || PIDS=""
echo "[phase6_full] PIDs: $LAUNCHER_PID $PIDS"
ps -o pid,comm -A 2>/dev/null | awk '$2=="VLLM::Worker" || $2=="VLLM::EngineCor" || $2=="VLLM::Worker_TP" {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
kill -9 $LAUNCHER_PID 2>/dev/null

echo "[phase6_full] $(TZ=Asia/Seoul date -Iseconds) DONE"

echo ""
echo "===== throughput stats ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" \
    | awk '{sum+=$1; if($1>max||NR==1)max=$1; if($1<min||NR==1)min=$1; n++} END {if (n>0) printf "n=%d avg=%.1f min=%.1f max=%.1f\n", n, sum/n, min, max}'
echo ""
echo "===== PROFILE PER-LAYER samples (log_freq=50) ====="
grep -c "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null
echo "----"
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | head -2
echo "..."
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== gpu_avg vs cdec_wait_avg ratio (cdec / gpu) ====="
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "ratio=[0-9.]+x" | sed 's/ratio=//;s/x//' \
    | awk '{sum+=$1; n++} END {if(n>0) printf "avg ratio = %.2fx\n", sum/n}'
echo ""
echo "===== SWAP_OUT / SWAP_IN per-call ms 통계 ====="
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++} END {if(n>0) printf "SWAP_OUT n=%d avg=%.2f ms\n", n, sum/n}'
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++} END {if(n>0) printf "SWAP_IN n=%d avg=%.2f ms\n", n, sum/n}'
echo ""
echo "===== crash ====="
grep -ciE "died unexpectedly|EngineDeadError" "${LOG_FILE}.stdout" 2>/dev/null
echo "crash"
echo "[phase6_full] analysis done"
