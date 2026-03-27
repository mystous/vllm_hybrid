#!/usr/bin/env bash
# =============================================================================
# serve.sh — vLLM server startup script
# Usage:
#   ./serve.sh gpu_only → GPU-only server
#   ./serve.sh hybrid   → Hybrid (GPU+CPU) server
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] .env file not found: $ENV_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

MODE="${1:-gpu_only}"
if [[ "$MODE" != "gpu_only" && "$MODE" != "hybrid" ]]; then
    echo "[ERROR] MODE must be 'gpu_only' or 'hybrid'." >&2
    exit 1
fi

# Set environment variables
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "============================================================"
echo " vLLM server starting: MODE=${MODE}"
echo " MODEL=${MODEL}"
echo " PORT=${PORT}"
echo "============================================================"

TP="${TENSOR_PARALLEL_SIZE:-1}"
TP_ARGS=""
if [[ "$TP" -gt 1 ]]; then
    TP_ARGS="--tensor-parallel-size ${TP}"
fi

if [[ "$MODE" == "gpu_only" ]]; then
    # shellcheck disable=SC2086
    exec python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
        ${TP_ARGS} \
        --disable-log-requests

elif [[ "$MODE" == "hybrid" ]]; then
    NUMA_FLAG="--hybrid-numa-aware"
    if [[ "${HYBRID_NUMA_AWARE,,}" == "false" ]]; then
        NUMA_FLAG="--no-hybrid-numa-aware"
    fi

    # If 0, use auto (--hybrid-cpu-* 0 lets _resolve_cpu_params auto-detect)
    CPU_MAX_SEQS_ARG=""
    CPU_KVCACHE_ARG=""
    CPU_THREADS_ARG=""
    CPU_ENGINES_ARG=""
    if [[ "${HYBRID_CPU_MAX_SEQS:-0}" -gt 0 ]]; then
        CPU_MAX_SEQS_ARG="--hybrid-cpu-max-seqs ${HYBRID_CPU_MAX_SEQS}"
    fi
    if [[ "${HYBRID_CPU_KVCACHE_GB:-0}" -gt 0 ]]; then
        CPU_KVCACHE_ARG="--hybrid-cpu-kvcache-gb ${HYBRID_CPU_KVCACHE_GB}"
    fi
    if [[ "${HYBRID_CPU_THREADS:-0}" -gt 0 ]]; then
        CPU_THREADS_ARG="--hybrid-cpu-threads ${HYBRID_CPU_THREADS}"
    fi
    if [[ "${HYBRID_NUM_CPU_ENGINES:-1}" -gt 1 ]]; then
        CPU_ENGINES_ARG="--hybrid-num-cpu-engines ${HYBRID_NUM_CPU_ENGINES}"
    fi

    # shellcheck disable=SC2086
    exec python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
        ${TP_ARGS} \
        --hybrid-mode parallel-batch \
        ${CPU_MAX_SEQS_ARG} \
        ${CPU_KVCACHE_ARG} \
        ${CPU_THREADS_ARG} \
        ${CPU_ENGINES_ARG} \
        --hybrid-routing-strategy "${HYBRID_ROUTING_STRATEGY}" \
        --hybrid-stats-log-interval "${HYBRID_STATS_LOG_INTERVAL}" \
        ${NUMA_FLAG} \
        --disable-log-requests
fi
