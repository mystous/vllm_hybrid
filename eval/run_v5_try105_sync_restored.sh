#!/usr/bin/env bash
# [try105] .cpu() sync 복원 후 v1.4 env (chain firing 98.7%) 측정.
# 목적: try104 crash (silent SEGV) 회피 + chain firing 빈번 발화 영역에서
# v1.5 의 다른 변경 (log/measurement 제거 + Option C v2 + Option O2 v2) 의
# 정량 효과 검증.
# 다른 모든 코드 변경 그대로. attention.py 의 seq_lens.cpu() sync 만 복원.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try105_sync_restored"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === v1.4 (try84) env — chain firing 98.7% 발화 영역 ===
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

echo "[try105] starting → ${LOG_FILE}"
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[try105] launcher PID=${LAUNCHER_PID}"

# 30분 측정
sleep 1800

WORKER_PIDS=$(ps -ef | grep "VLLM::Worker" | grep -v grep | awk '{print $2}')
echo "[try105] worker PIDs: $WORKER_PIDS"

pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[try105] DONE $(date -Iseconds)"

echo ""
echo "===== throughput (Avg gen, last 5) ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo ""
echo "===== Processed prompts ====="
grep -oE 'Processed prompts:[^,]*' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== crash ====="
grep -iE 'error|exception|traceback|engine dead|died unexpectedly' "${LOG_FILE}.stdout" 2>/dev/null | head -5
