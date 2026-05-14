#!/usr/bin/env bash
# [Phase E19] ncu (Nsight Compute) — chain firing 영역 kernel detail.
# launch-skip 5000 (warmup 영역 건너뛰기) + launch-count 30 (30 kernel sample).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="E19_ncu"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# v1.5 env
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
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE VLLM_DEBUG_CDEC_PATH

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[E19-ncu] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

# ncu launch-skip 5000 + launch-count 30
# --set full → 모든 metric. --section LaunchStats/Occupancy/MemoryWorkloadAnalysis
taskset -c 0-111 /usr/local/cuda/bin/ncu \
    --target-processes all \
    --launch-skip 5000 \
    --launch-count 30 \
    --section LaunchStats \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStats \
    --export "${OUT_DIR}/ncu_report" \
    "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 50 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[E19-ncu] launcher PID=${LAUNCHER_PID}"

# ncu 영역은 매우 느림. 600초.
sleep 600

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null
pgrep -f "ncu\|run_neo_baseline" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[E19-ncu] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== ncu output files ====="
ls -la "${OUT_DIR}/" 2>/dev/null
echo ""
echo "===== ncu report summary (top 10 kernels) ====="
if [ -f "${OUT_DIR}/ncu_report.ncu-rep" ]; then
    /usr/local/cuda/bin/ncu --import "${OUT_DIR}/ncu_report.ncu-rep" --csv 2>/dev/null | head -20
fi
echo "[E19-ncu] done"
