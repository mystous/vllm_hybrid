#!/bin/bash
# Step 1 server launcher — supports multi-instance concurrent runs.
# Args: PROFILE PORT GPU CPU_THREADS [PIN_CORES]
#  PROFILE     = vanilla | offload
#  PORT        = e.g. 8011..8018
#  GPU         = 0..7
#  CPU_THREADS = kt-kernel CPUINFER threads (offload only); ignored for vanilla
#  PIN_CORES   = optional CPU set for taskset (e.g. "0-27"); empty = no pin
set -uo pipefail
PROFILE=${1:-vanilla}
PORT=${2:-8011}
GPU=${3:-0}
CPU_THREADS=${4:-28}
PIN_CORES=${5:-}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs/step1"
mkdir -p "$LOGS"

MODEL_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"

export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_V1=1

if [ "$PROFILE" = "offload" ]; then
    export VLLM_MOE_CPU_OFFLOAD=1
    export VLLM_MOE_NUM_GPU_EXPERTS=32
    export VLLM_MOE_CPUINFER_THREADS=$CPU_THREADS
    export VLLM_MOE_THREADPOOL_COUNT=2
    export VLLM_MOE_KT_METHOD=${VLLM_MOE_KT_METHOD:-BF16}
    export VLLM_MOE_KT_WEIGHT_PATH="$MODEL_PATH"
    export VLLM_MOE_KT_SITE_PACKAGES=/workspace/sglang_kt_prj/lib/python3.12/site-packages
    export VLLM_MOE_KT_DEBUG=${VLLM_MOE_KT_DEBUG:-0}
    export VLLM_MOE_KT_DEBUG_SYNC=${VLLM_MOE_KT_DEBUG_SYNC:-0}
    export VLLM_MOE_NO_CPU_STREAM=${VLLM_MOE_NO_CPU_STREAM:-1}
    export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
else
    export VLLM_MOE_CPU_OFFLOAD=0
fi

LOG="$LOGS/${PROFILE}_gpu${GPU}_p${PORT}.log"
PIDFILE="$LOGS/${PROFILE}_gpu${GPU}_p${PORT}.pid"
echo "[$(date -Iseconds)] starting profile=$PROFILE port=$PORT gpu=$GPU threads=$CPU_THREADS pin=$PIN_CORES log=$LOG"

KERNEL_CFG='{"moe_backend":"triton","enable_flashinfer_autotune":false}'

CMD=(/workspace/vllm_dev_prj/bin/vllm serve "$MODEL_PATH"
    --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --gpu-memory-utilization 0.85
    --max-model-len 4096
    --enforce-eager
    --kernel-config "$KERNEL_CFG"
    --served-model-name moe-offload-test)

if [ -n "$PIN_CORES" ]; then
    setsid taskset -c "$PIN_CORES" "${CMD[@]}" > "$LOG" 2>&1 &
else
    setsid "${CMD[@]}" > "$LOG" 2>&1 &
fi
echo $! > "$PIDFILE"
echo "PID=$(cat $PIDFILE) PGID=$(ps -o pgid= -p $(cat $PIDFILE) | tr -d ' ')"
