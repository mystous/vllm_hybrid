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
#
# PROFILE is the master switch for hybrid observability.
#   VLLM_HYBRID_PROFILE=0           — all HYBRID trace/profile/info markers down
#   VLLM_HYBRID_PROFILE=1           — hybrid observability on
#   VLLM_HYBRID_PROFILE_EVERY=N     — breakdown emit interval
#                                     (default 10 if PROFILE=1, 0 otherwise)
#
# TRACE is subordinate to PROFILE and only affects coarse per-step sampling.
#   VLLM_HYBRID_TRACE=1             — log every CPU exec_model + attn call
#   VLLM_HYBRID_TRACE_EVERY=N (N>0) — log every N-th call at INFO
export VLLM_HYBRID_TRACE="${VLLM_HYBRID_TRACE:-0}"
export VLLM_HYBRID_TRACE_EVERY="${VLLM_HYBRID_TRACE_EVERY:-0}"
export VLLM_HYBRID_PROFILE="${VLLM_HYBRID_PROFILE:-0}"
export VLLM_HYBRID_PROFILE_EVERY="${VLLM_HYBRID_PROFILE_EVERY:-0}"
export VLLM_HYBRID_PROFILE_SUBLAYER="${VLLM_HYBRID_PROFILE_SUBLAYER:-0}"

# NinjaGap §03 Phase 2: 1GB hugetlbfs (기본 off — 호스트 사전 준비 필요)
#   호스트 prep:
#     echo 64 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
#     mkdir -p /mnt/hugetlb_1g
#     mount -t hugetlbfs -o pagesize=1G,size=64G none /mnt/hugetlb_1g
export HYBRID_HUGETLB_1G_ENABLE="${HYBRID_HUGETLB_1G_ENABLE:-0}"
export HYBRID_HUGETLB_1G_PATH="${HYBRID_HUGETLB_1G_PATH:-/mnt/hugetlb_1g}"
# Commit 2 flag — 현재는 no-op (경고 로그만). slab allocator 구현 후 활성화
export HYBRID_HUGETLB_1G_BIND_WEIGHTS="${HYBRID_HUGETLB_1G_BIND_WEIGHTS:-0}"

echo "============================================================"
echo " vLLM server starting: MODE=${MODE}"
echo " MODEL=${MODEL}"
echo " PORT=${PORT}"
echo " ENV_FILE=${ENV_FILE}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────
# 서버 stdout/stderr 를 고정 위치 로그 파일에 tee 로 복제해서 bench.sh
# 가 런 완료 후 이 파일을 results/ 디렉토리에 복사할 수 있게 함.
# 경로: eval/serve_logs/server_latest.log (symlink + 타임스탬프)
# bench.sh 는 이 로그를 RUN_DIR 로 복사.
# ─────────────────────────────────────────────────────────────────────
SERVER_LOG_DIR="${SCRIPT_DIR}/serve_logs"
mkdir -p "${SERVER_LOG_DIR}"
SERVER_LOG_FILE="${SERVER_LOG_DIR}/server_$(date +%Y%m%d_%H%M%S)_${MODE}.log"
# latest 심링크 갱신
ln -sf "${SERVER_LOG_FILE}" "${SERVER_LOG_DIR}/server_latest.log"
echo " SERVER_LOG=${SERVER_LOG_FILE}"

# 서버 내부 manifest 는 임시 디렉토리에 저장. bench.sh 가 최종 RUN_DIR 결정하고
# 이 임시 디렉토리에서 manifest 를 복사한다 (기존 server log slice 방식과 동일).
# VLLM_HYBRID_RESULT_DIR 은 [HYBRID-APPLIED-FEATURES] manifest JSON 저장 경로.
if [[ "${VLLM_HYBRID_PROFILE}" == "1" ]]; then
    export VLLM_HYBRID_RESULT_DIR="${VLLM_HYBRID_RESULT_DIR:-${SERVER_LOG_DIR}/profile_latest}"
    mkdir -p "${VLLM_HYBRID_RESULT_DIR}"
    env | grep -E '^(HYBRID_|VLLM_HYBRID_)' | sort \
        > "${VLLM_HYBRID_RESULT_DIR}/env_snapshot.txt"
    (cd "${SCRIPT_DIR}/.." && git rev-parse HEAD 2>/dev/null || true) \
        > "${VLLM_HYBRID_RESULT_DIR}/git_sha.txt"
    echo " PROFILE_MODE=on, manifest staging dir: ${VLLM_HYBRID_RESULT_DIR}"
    echo "   (bench.sh 가 최종 measurement_results/<HW>/g0_<NN>/seqs<N>/ 로 결정)"
else
    export VLLM_HYBRID_RESULT_DIR="${VLLM_HYBRID_RESULT_DIR:-${SERVER_LOG_DIR}/profile_latest}"
fi
echo "============================================================"

TP="${TENSOR_PARALLEL_SIZE:-1}"
TP_ARGS=""
if [[ "${TP}" -gt 1 ]]; then
    TP_ARGS="--tensor-parallel-size ${TP}"
fi

if [[ "${MODE}" == "gpu_only" ]]; then
    # shellcheck disable=SC2086
    # tee 로 stdout/stderr 를 SERVER_LOG_FILE 에 복제.
    # exec 을 포기하고 pipe 를 쓰므로 trap 으로 자식 python 종료 보장.
    python -u -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
        ${TP_ARGS} \
        ${EXTRA_SERVE_ARGS:-} \
        --disable-log-requests 2>&1 | tee "${SERVER_LOG_FILE}"

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
    # HYBRID_NUM_CPU_ENGINES: 0=auto(NUMA 수), 1=single, 2+=명시
    # env 에 설정되어 있으면 항상 CLI 로 전달 (0 포함). 미설정이면 생략 → argparse default=0(auto)
    if [[ -n "${HYBRID_NUM_CPU_ENGINES:-}" ]]; then
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
    python -u -m vllm.entrypoints.openai.api_server \
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
        --disable-log-requests 2>&1 | tee "${SERVER_LOG_FILE}"
fi
