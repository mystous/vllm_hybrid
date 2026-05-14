#!/usr/bin/env bash
# NEO standard run — TSK_019 v1.7 SUB_027 winning config.
# H5 측정 결과 (2026-05-14, eval/results/20260514_005321_hyp500_H5_gmu_mirror80/):
#   500p × 8192 in/out, Llama-70B TP=8 H100×8
#   output_tps = 2,302.0 (vanilla 4,682.1 의 49.2%, NEO sync try102 627.6 대비 +267%)
#
# 표준 설정:
#   - gpu_memory_utilization=0.92 (default 0.85 에서 상향, KV blocks +11.6%)
#   - VLLM_NEO_MIRROR_MAX=80 (코드 default 도 56 → 80)
#   - VLLM_NEO_ASYNC_SWAP_BUFFERS=3 (staging buffer pool, worker당 1.92 GiB pinned)
#   - VLLM_NEO_CPU_RESIDENT_REQS=128 (default 유지, NeoCpuKvBuffer 한계 정합)
#   - VLLM_NEO_PROFILE=1 (측정 ON)
#
# 보존 영역: hardware tuning (CPU pin / NUMA / OMP).
# 제거 영역: 모든 실험적 NEO env (Option A/C/K/L/M2/O2, 강제 swap, force pipelined, etc.)
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="neo_standard"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === 실험적 NEO env 모두 unset ===
unset VLLM_NEO_OPTION_A VLLM_NEO_OPTION_C VLLM_NEO_OPTION_K
unset VLLM_NEO_OPTION_L VLLM_NEO_OPTION_M2 VLLM_NEO_OPTION_O2
unset VLLM_NEO_OPTION_C_FULL_MIRROR
unset VLLM_NEO_SWAP_OUT_RATIO VLLM_NEO_PREDICTIVE_THRESHOLD
unset VLLM_NEO_LOAD_AWARE_MIN_RUNNING
unset VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP
unset VLLM_NEO_MAX_SWAP_IN_PER_STEP
unset VLLM_NEO_MIRROR_MIN_BUFFER VLLM_NEO_MIN_RUNNING_DECODE
unset VLLM_NEO_SWAP_COOLDOWN
unset VLLM_NEO_FORCE_SWAP_IN VLLM_NEO_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_NEOSCHED_STEP23 VLLM_NEO_DRIVE_6STEP VLLM_NEO_6STEP_DRY_RUN
unset VLLM_NEO_DECIDE_MODE_BALANCE
unset VLLM_NEO_HEURISTIC_LINR_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_PREF_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_GDEC_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_CDEC_PER_TOKEN_PAIR_MS
unset VLLM_NEO_HEURISTIC_LNCH_MS
unset VLLM_NEO_SWAP_IN_ORDER
unset VLLM_NEO_ASYNC_CDEC VLLM_NEO_CDEC_PIPELINE_DEPTH
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH
unset ENABLE_NEO_INV

# === 표준 NEO env (SUB_027 winning) ===
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_CPU_RESIDENT_REQS=128   # default, NeoCpuKvBuffer 한계 정합
export VLLM_NEO_ASYNC_SWAP_BUFFERS=3    # staging buffer pool size (SUB_026)
# VLLM_NEO_MIRROR_MAX 는 코드 default 80 사용 (이전 56, SUB_027 상향)
export VLLM_NEO_PROFILE=1               # PROFILE 로그 활성

# === 환경 영역 — hardware tuning (NUMA + OMP) ===
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

echo "[neo_standard] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[neo_standard] config: gpu_util=0.92 mirror_max=80 (code default) async_buffers=3"
echo "[neo_standard] active VLLM_NEO_* env:"
env | grep -E "^VLLM_NEO_" | sort
echo ""

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.92 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[neo_standard] launcher PID=${LAUNCHER_PID}"

# Polling 방식 — `wait` builtin 이 nested shell 에서 즉시 127 반환 회피.
START_TS=$(date +%s)
MAX_WAIT=5400  # 90min safety cap
while kill -0 ${LAUNCHER_PID} 2>/dev/null; do
    sleep 15
    NOW_TS=$(date +%s)
    if [ $((NOW_TS - START_TS)) -gt ${MAX_WAIT} ]; then
        echo "[neo_standard] timeout — killing launcher"
        kill -9 ${LAUNCHER_PID} 2>/dev/null
        break
    fi
done
ELAPSED=$(($(date +%s) - START_TS))
echo "[neo_standard] launcher exited after ${ELAPSED}s"

pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[neo_standard] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== output_tps ====="
cat "${OUT_DIR}/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('output_tps=', d.get('output_tps'), 'wall_s=', d.get('generate_wall_s'))" 2>/dev/null
echo ""
echo "===== swap counts (TP0) ====="
echo "  async: $(grep 'PROFILE SWAP_OUT async' ${LOG_FILE}.stdout 2>/dev/null | grep -c Worker_TP0)"
echo "  sync:  $(grep 'PROFILE SWAP_OUT[^_]' ${LOG_FILE}.stdout 2>/dev/null | grep -c Worker_TP0)"
echo ""
echo "===== crash check ====="
echo "  crash: $(grep -ciE 'died unexpectedly|EngineDeadError' ${LOG_FILE}.stdout 2>/dev/null)"
echo "[neo_standard] analysis done"
