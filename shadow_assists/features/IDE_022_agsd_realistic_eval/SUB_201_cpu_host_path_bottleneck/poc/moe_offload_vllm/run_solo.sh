#!/bin/bash
# Solo (single-instance) benchmark — used for Step 2/3 (dual-stream, INT8).
# Args: PROFILE PORT GPU [LABEL]
# Env knobs (passed through to start_server.sh):
#   VLLM_MOE_NO_CPU_STREAM   (0=dual-stream, 1=serial; default 1)
#   VLLM_MOE_KT_METHOD       (BF16|AMXINT8)
#   VLLM_MOE_NUM_GPU_EXPERTS (default 32)
#   VLLM_MOE_CPUINFER_THREADS (default 112)
set -uo pipefail
PROFILE=${1:-offload}
PORT=${2:-8009}
GPU=${3:-0}
LABEL=${4:-default}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lifecycle.sh"

LOG_DIR="$HERE/logs/solo/${LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[solo] profile=$PROFILE port=$PORT gpu=$GPU label=$LABEL log_dir=$LOG_DIR"

# Copy env knobs (start_server.sh reads them).
export VLLM_MOE_NO_CPU_STREAM=${VLLM_MOE_NO_CPU_STREAM:-1}
export VLLM_MOE_KT_METHOD=${VLLM_MOE_KT_METHOD:-BF16}
export VLLM_MOE_NUM_GPU_EXPERTS=${VLLM_MOE_NUM_GPU_EXPERTS:-32}
export VLLM_MOE_CPUINFER_THREADS=${VLLM_MOE_CPUINFER_THREADS:-112}
export VLLM_MOE_THREADPOOL_COUNT=${VLLM_MOE_THREADPOOL_COUNT:-2}

bash "$HERE/start_server.sh" "$PROFILE" "$PORT" "$GPU"
PIDFILE="$HERE/logs/${PROFILE}_server.pid"

if ! wait_ready "$PORT" 900; then
    echo "[solo] server failed to become ready"
    kill_pgroup "$PIDFILE"
    kill_gpu_orphans "$GPU"
    exit 1
fi

top -bn1 | head -5 > "$LOG_DIR/cpu_before.txt" 2>&1 || true

LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib \
    /workspace/vllm_dev_prj/bin/python "$HERE/measure_client.py" \
    --url "http://127.0.0.1:${PORT}/v1" \
    --model moe-offload-test --n 100 --conc 8 --max-tokens 256 \
    --out "$LOG_DIR/result.json" 2>&1 | tee "$LOG_DIR/measure.log"

top -bn1 | head -5 > "$LOG_DIR/cpu_after.txt" 2>&1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_after.txt" 2>&1 || true

# Tear down
kill_pgroup "$PIDFILE"
kill_gpu_orphans "$GPU"
wait_gpu_free "$GPU" 60 || true
echo "[solo] done. results in $LOG_DIR"
