#!/usr/bin/env bash
# =============================================================================
# serve.sh — vLLM server startup script
# Usage:
#   ./serve.sh <mode> [env_file]
#   mode    : gpu_only | hybrid
#   env_file: path to .env config (optional if EVAL_ENV_FILE is set)
#
# Examples:
#   EVAL_ENV_FILE=env/h100x8.env ./serve.sh hybrid
#   ./serve.sh gpu_only env/dev_rtx3090.env
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve env file: arg > EVAL_ENV_FILE env var > default .env
if [[ -n "${2:-}" ]]; then
    ENV_ARG="$2"
    if [[ ! "${ENV_ARG}" = /* ]]; then
        ENV_ARG="${SCRIPT_DIR}/${ENV_ARG}"
    fi
    ENV_FILE="${ENV_ARG}"
elif [[ -n "${EVAL_ENV_FILE:-}" ]]; then
    ENV_FILE="${EVAL_ENV_FILE}"
else
    ENV_FILE="${SCRIPT_DIR}/.env"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] env file not found: ${ENV_FILE}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

MODE="${1:-gpu_only}"
if [[ "${MODE}" != "gpu_only" && "${MODE}" != "hybrid" ]]; then
    echo "[ERROR] MODE must be 'gpu_only' or 'hybrid'." >&2
    exit 1
fi

# Set environment variables
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Hybrid debugging knobs (consumed by hybrid_core/cpu_worker/cpu_attn)
# Default hybrid trace: silent during serving. Boot/shutdown markers are
# emitted via logger.info regardless. To turn trace on for smoke/debug:
#   VLLM_HYBRID_TRACE=1             — log on every CPU exec_model + attn call
#   VLLM_HYBRID_TRACE_EVERY=N (N>0) — log every N-th call at INFO
export VLLM_HYBRID_TRACE="${VLLM_HYBRID_TRACE:-0}"
export VLLM_HYBRID_TRACE_EVERY="${VLLM_HYBRID_TRACE_EVERY:-0}"

echo "============================================================"
echo " vLLM server starting: MODE=${MODE}"
echo " MODEL=${MODEL}"
echo " PORT=${PORT}"
echo " ENV_FILE=${ENV_FILE}"
echo "============================================================"

TP="${TENSOR_PARALLEL_SIZE:-1}"
TP_ARGS=""
if [[ "${TP}" -gt 1 ]]; then
    TP_ARGS="--tensor-parallel-size ${TP}"
fi

if [[ "${MODE}" == "gpu_only" ]]; then
    # shellcheck disable=SC2086
    exec python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
        ${TP_ARGS} \
        ${EXTRA_SERVE_ARGS:-} \
        --disable-log-requests

elif [[ "${MODE}" == "hybrid" ]]; then
    NUMA_FLAG="--hybrid-numa-aware"
    if [[ "${HYBRID_NUMA_AWARE,,}" == "false" ]]; then
        NUMA_FLAG="--no-hybrid-numa-aware"
    fi

    # If 0, use auto (--hybrid-cpu-* 0 lets _resolve_cpu_params auto-detect)
    CPU_MAX_SEQS_ARG=""
    CPU_KVCACHE_ARG=""
    CPU_THREADS_ARG=""
    CPU_CORE_RATIO_ARG=""
    CPU_ENGINES_ARG=""
    CPU_MAX_BATCHED_TOKENS_ARG=""
    CPU_PREFILL_THRESHOLD_ARG=""
    WARMUP_REQUESTS_ARG=""
    ROUTING_PRIORITY_ARG=""
    if [[ "${HYBRID_CPU_MAX_SEQS:-0}" -gt 0 ]]; then
        CPU_MAX_SEQS_ARG="--hybrid-cpu-max-seqs ${HYBRID_CPU_MAX_SEQS}"
    fi
    if [[ "${HYBRID_CPU_KVCACHE_GB:-0}" -gt 0 ]]; then
        CPU_KVCACHE_ARG="--hybrid-cpu-kvcache-gb ${HYBRID_CPU_KVCACHE_GB}"
    fi
    if [[ "${HYBRID_CPU_THREADS:-0}" -gt 0 ]]; then
        CPU_THREADS_ARG="--hybrid-cpu-threads ${HYBRID_CPU_THREADS}"
    fi
    if [[ -n "${HYBRID_CPU_CORE_RATIO:-}" ]]; then
        CPU_CORE_RATIO_ARG="--hybrid-cpu-core-ratio ${HYBRID_CPU_CORE_RATIO}"
    fi
    if [[ "${HYBRID_NUM_CPU_ENGINES:-1}" -gt 1 ]]; then
        CPU_ENGINES_ARG="--hybrid-num-cpu-engines ${HYBRID_NUM_CPU_ENGINES}"
    fi
    if [[ "${HYBRID_CPU_MAX_BATCHED_TOKENS:-0}" -gt 0 ]]; then
        CPU_MAX_BATCHED_TOKENS_ARG="--hybrid-cpu-max-batched-tokens ${HYBRID_CPU_MAX_BATCHED_TOKENS}"
    fi
    if [[ "${HYBRID_CPU_PREFILL_THRESHOLD:-0}" -gt 0 ]]; then
        CPU_PREFILL_THRESHOLD_ARG="--hybrid-cpu-prefill-threshold ${HYBRID_CPU_PREFILL_THRESHOLD}"
    fi
    if [[ "${HYBRID_WARMUP_REQUESTS:-0}" -gt 0 ]]; then
        WARMUP_REQUESTS_ARG="--hybrid-warmup-requests ${HYBRID_WARMUP_REQUESTS}"
    fi
    if [[ -n "${HYBRID_ROUTING_PRIORITY:-}" && "${HYBRID_ROUTING_PRIORITY}" != "gpu-first" ]]; then
        ROUTING_PRIORITY_ARG="--hybrid-routing-priority ${HYBRID_ROUTING_PRIORITY}"
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
        ${CPU_CORE_RATIO_ARG} \
        ${CPU_ENGINES_ARG} \
        ${CPU_MAX_BATCHED_TOKENS_ARG} \
        ${CPU_PREFILL_THRESHOLD_ARG} \
        ${WARMUP_REQUESTS_ARG} \
        ${ROUTING_PRIORITY_ARG} \
        --hybrid-routing-strategy "${HYBRID_ROUTING_STRATEGY}" \
        --hybrid-stats-log-interval "${HYBRID_STATS_LOG_INTERVAL}" \
        ${NUMA_FLAG} \
        ${EXTRA_SERVE_ARGS:-} \
        --disable-log-requests
fi
