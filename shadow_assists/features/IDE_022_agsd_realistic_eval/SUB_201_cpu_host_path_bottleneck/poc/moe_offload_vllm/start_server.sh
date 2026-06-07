#!/bin/bash
# Start vllm server. Args: PROFILE PORT GPU
# PROFILE = vanilla | offload
set -uo pipefail
PROFILE=${1:-vanilla}
PORT=${2:-8009}
GPU=${3:-0}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

MODEL_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"

export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_V1=1
# Force triton MoE backend (no fancy specialized kernels) to keep behavior comparable
# to SGLang baseline (which used triton attention + triton moe-runner).

if [ "$PROFILE" = "offload" ]; then
    export VLLM_MOE_CPU_OFFLOAD=1
    export VLLM_MOE_NUM_GPU_EXPERTS=${VLLM_MOE_NUM_GPU_EXPERTS:-32}
    export VLLM_MOE_CPUINFER_THREADS=${VLLM_MOE_CPUINFER_THREADS:-112}
    export VLLM_MOE_THREADPOOL_COUNT=${VLLM_MOE_THREADPOOL_COUNT:-2}
    export VLLM_MOE_KT_METHOD=${VLLM_MOE_KT_METHOD:-BF16}
    export VLLM_MOE_KT_WEIGHT_PATH="${VLLM_MOE_KT_WEIGHT_PATH:-$MODEL_PATH}"
    export VLLM_MOE_KT_SITE_PACKAGES=${VLLM_MOE_KT_SITE_PACKAGES:-/workspace/sglang_kt_prj/lib/python3.12/site-packages}
    export VLLM_MOE_KT_DEBUG=${VLLM_MOE_KT_DEBUG:-0}
    export VLLM_MOE_KT_DEBUG_SYNC=${VLLM_MOE_KT_DEBUG_SYNC:-0}
    # Default: submit + sync on main_stream (no separate cpu_stream).
    # Mirrors SGLang's SGLANG_KT_HYBRID_NO_CPU_STREAM=1 fallback. Once boot is
    # stable, set VLLM_MOE_NO_CPU_STREAM=0 to enable the dual-stream path.
    export VLLM_MOE_NO_CPU_STREAM=${VLLM_MOE_NO_CPU_STREAM:-1}
    export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
else
    export VLLM_MOE_CPU_OFFLOAD=0
fi

LOG="$LOGS/${PROFILE}_server.log"
PIDFILE="$LOGS/${PROFILE}_server.pid"
echo "[$(date -Iseconds)] starting profile=$PROFILE port=$PORT gpu=$GPU  log=$LOG"

# Disable flashinfer autotune for the offload profile — the autotune phase
# re-runs _dummy_run after profile_run and we have not yet stabilised every
# kt-path code path. moe_backend=triton keeps FlashInfer's monolithic MoE
# kernel out of the way so our forward_cuda hook fires.
KERNEL_CFG='{"moe_backend":"triton","enable_flashinfer_autotune":false}'

setsid /workspace/vllm_dev_prj/bin/vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enforce-eager \
    --kernel-config "$KERNEL_CFG" \
    --served-model-name moe-offload-test \
    > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "PID=$(cat $PIDFILE) PGID=$(ps -o pgid= -p $(cat $PIDFILE) | tr -d ' ')"
