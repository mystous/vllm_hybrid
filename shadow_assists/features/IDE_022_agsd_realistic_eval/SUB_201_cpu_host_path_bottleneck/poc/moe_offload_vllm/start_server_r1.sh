#!/bin/bash
# R1 (DeepSeek-R1 671B) server. Args: PROFILE PORT TP
#  PROFILE = vanilla | offload (offload uses BF16 kt-kernel on CPU)
#  PORT
#  TP      = tensor parallel size (default 8)
set -uo pipefail
PROFILE=${1:-vanilla}
PORT=${2:-8019}
TP=${3:-8}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs/r1"
mkdir -p "$LOGS"

MODEL_PATH="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"

export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib
export VLLM_USE_V1=1
# DeepGEMM FP8 weight post-processing on B200 crashes with
# "Cannot access data pointer of Tensor that doesn't have storage" — disable.
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_USE_DEEP_GEMM_E8M0=${VLLM_USE_DEEP_GEMM_E8M0:-0}
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES:-0}
# FA4 on B200 imports cutlass.pipeline.PipelineClcFetchAsync which is missing
# in the shipped CUTLASS DSL — force FA2 fallback via batch_invariant flag
# (fa_utils.py:117 forces fa_version=2 if VLLM_BATCH_INVARIANT and fa_version==4).
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}

if [ "$PROFILE" = "offload" ]; then
    export VLLM_MOE_CPU_OFFLOAD=1
    export VLLM_MOE_NUM_GPU_EXPERTS=${VLLM_MOE_NUM_GPU_EXPERTS:-64}
    export VLLM_MOE_CPUINFER_THREADS=${VLLM_MOE_CPUINFER_THREADS:-112}
    export VLLM_MOE_THREADPOOL_COUNT=${VLLM_MOE_THREADPOOL_COUNT:-2}
    export VLLM_MOE_KT_METHOD=${VLLM_MOE_KT_METHOD:-BF16}
    export VLLM_MOE_KT_WEIGHT_PATH="${VLLM_MOE_KT_WEIGHT_PATH:-$MODEL_PATH}"
    export VLLM_MOE_KT_SITE_PACKAGES=${VLLM_MOE_KT_SITE_PACKAGES:-/workspace/sglang_kt_prj/lib/python3.12/site-packages}
    export VLLM_MOE_NO_CPU_STREAM=${VLLM_MOE_NO_CPU_STREAM:-1}
else
    export VLLM_MOE_CPU_OFFLOAD=0
fi

LOG="$LOGS/${PROFILE}_tp${TP}.log"
PIDFILE="$LOGS/${PROFILE}_tp${TP}.pid"
echo "[$(date -Iseconds)] starting R1 profile=$PROFILE port=$PORT tp=$TP log=$LOG"

KERNEL_CFG='{"moe_backend":"triton","enable_flashinfer_autotune":false}'
# Force TRITON_MLA backend AND flash_attn_version=3 to fully avoid the cute
# (FA4) import path that breaks with the shipped CUTLASS DSL.
ATTN_CFG='{"backend":"TRITON_MLA","flash_attn_version":3}'

setsid /workspace/vllm_dev_prj/bin/vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 \
    --max-model-len 4096 \
    --enforce-eager \
    --kernel-config "$KERNEL_CFG" \
    --attention-config "$ATTN_CFG" \
    --trust-remote-code \
    --served-model-name r1-offload-test \
    > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "PID=$(cat $PIDFILE) PGID=$(ps -o pgid= -p $(cat $PIDFILE) | tr -d ' ')"
