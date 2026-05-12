#!/usr/bin/env bash
# [Phase 6.2 / 6P] NEO PROFILE — cdec dispatch + swap_in/out component time.
# 기존 코드 적재된 영역 (attention.py:1030-1116, gpu_model_runner.py:6406-6510)
# 의 VLLM_NEO_PROFILE=1 env 만 활성. 5분 short measurement.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="neo_profile_component"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === try105 env (chain firing 95.6%) ===
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

# === VLLM_NEO_PROFILE 활성 (분석 단계 한정) ===
export VLLM_NEO_PROFILE=1

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[neo_profile] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[neo_profile] VLLM_NEO_PROFILE=${VLLM_NEO_PROFILE}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[neo_profile] launcher PID=${LAUNCHER_PID}"

# 5분 measurement (4분 engine init + 1분 chain firing 영역)
sleep 300

# Cleanup
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[neo_profile] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -5
echo ""
echo "===== PROFILE PER-LAYER (attention.py — cdec wait + GPU forward component time) ====="
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo ""
echo "===== PROFILE SWAP_OUT (gpu_model_runner.py — swap_out per-call ms) ====="
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "..."
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "total SWAP_OUT events: $(grep -c "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== PROFILE SWAP_IN (gpu_model_runner.py — swap_in per-call ms) ====="
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "..."
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "total SWAP_IN events: $(grep -c "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== SWAP_OUT elapsed_ms 통계 ====="
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++; if($1>max||NR==1)max=$1; if($1<min||NR==1)min=$1} END {if(n>0) printf "n=%d avg=%.2f min=%.2f max=%.2f ms\n", n, sum/n, min, max}'
echo ""
echo "===== SWAP_IN elapsed_ms 통계 ====="
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++; if($1>max||NR==1)max=$1; if($1<min||NR==1)min=$1} END {if(n>0) printf "n=%d avg=%.2f min=%.2f max=%.2f ms\n", n, sum/n, min, max}'
echo ""
echo "===== crash check ====="
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[neo_profile] analysis done"
