#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — GPU-only / Hybrid full evaluation pipeline
#
# Execution order:
#   1. Start GPU-only server
#   2. Start monitor (GPU/CPU utilization)
#   3. Wait for server to be ready
#   4. Run benchmark
#   5. Stop monitor, stop server
#   6. Start Hybrid server
#   7. Start monitor
#   8. Wait for server to be ready
#   9. Run benchmark
#  10. Stop monitor, stop server
#  11. Generate comparison report
#
# Usage:
#   ./run_eval.sh <env_file> [mode]
#
#   env_file: path to .env config (e.g. env/h100x8.env)
#   mode    : all | gpu_only | hybrid | compare  (default: all)
#
# Examples:
#   ./run_eval.sh env/dev_rtx3090.env
#   ./run_eval.sh env/h100x8.env hybrid
#   ./run_eval.sh env/h100x8.env compare
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <env_file> [mode]" >&2
    echo "  env_file: path to .env config (e.g. env/h100x8.env)" >&2
    echo "  mode    : all | gpu_only | hybrid | compare (default: all)" >&2
    echo "" >&2
    echo "Available env files:" >&2
    for f in "${SCRIPT_DIR}"/envs/*.env; do
        echo "  ${f##"${SCRIPT_DIR}/"}" >&2
    done
    exit 1
fi

ENV_FILE="$1"
# Resolve relative path against script dir
if [[ ! "${ENV_FILE}" = /* ]]; then
    ENV_FILE="${SCRIPT_DIR}/${ENV_FILE}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] env file not found: ${ENV_FILE}" >&2
    exit 1
fi

# Export so sub-scripts (serve.sh, benchmark.sh) can inherit
export EVAL_ENV_FILE="${ENV_FILE}"

# shellcheck disable=SC1090
source "${ENV_FILE}"

RESULTS_BASE="${SCRIPT_DIR}/${RESULTS_DIR:-results}"

# Run timestamp — all outputs go into results/<RUN_TS>/
RUN_TS="${RUN_TS:-$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')}"
RESULTS_DIR="${RESULTS_BASE}/${RUN_TS}"
export EVAL_RUN_DIR="${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Keep a symlink results/latest → most recent run
ln -sfn "${RUN_TS}" "${RESULTS_BASE}/latest"

MODE="${2:-all}"  # all / gpu_only / hybrid / compare

SERVER_PID=""
MONITOR_PID=""

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local url="http://localhost:${PORT}/health"
    local timeout="${SERVER_READY_TIMEOUT:-300}"
    local poll="${SERVER_READY_POLL:-3}"
    local elapsed=0

    log "Waiting for server to be ready... (up to ${timeout}s)"
    while ! curl -sf "$url" > /dev/null 2>&1; do
        if [[ $elapsed -ge $timeout ]]; then
            log "[ERROR] Server startup timed out (exceeded ${timeout}s)"
            return 1
        fi
        sleep "$poll"
        elapsed=$((elapsed + poll))
        log "  Waiting... ${elapsed}/${timeout}s"
    done
    log "Server ready (took ${elapsed}s)"
}

start_server() {
    local server_mode="$1"
    log "=== Starting server: MODE=${server_mode} ==="
    bash "${SCRIPT_DIR}/serve.sh" "${server_mode}" \
        > "${RESULTS_DIR}/${server_mode}_serve.log" 2>&1 &
    SERVER_PID=$!
    log "Server PID: ${SERVER_PID}"
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping server (PID=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
    # Clean up any remaining vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
}

start_monitor() {
    local prefix="$1"
    local interval="${MONITOR_INTERVAL:-1}"
    log "Starting monitor: prefix=${prefix}"
    python3 "${SCRIPT_DIR}/monitor.py" "${prefix}" --interval "${interval}" \
        > "${RESULTS_DIR}/monitor_${prefix##*/}.log" 2>&1 &
    MONITOR_PID=$!
    log "Monitor PID: ${MONITOR_PID}"
}

stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        log "Stopping monitor (PID=${MONITOR_PID})"
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

cleanup() {
    log "Cleaning up..."
    stop_monitor
    stop_server
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# GPU-only evaluation
# ---------------------------------------------------------------------------

run_gpu_only() {
    log "=========================================="
    log " [1/2] Starting GPU-only evaluation"
    log "=========================================="

    stop_server   # Clean up any leftover server

    start_server "gpu_only"
    start_monitor "${RESULTS_DIR}/gpu_only_monitor"

    wait_for_server

    log "--- Running GPU-only benchmark ---"
    bash "${SCRIPT_DIR}/benchmark.sh" "gpu_only"

    stop_monitor
    stop_server

    log "GPU-only evaluation complete."
}

# ---------------------------------------------------------------------------
# Hybrid evaluation
# ---------------------------------------------------------------------------

run_hybrid() {
    log "=========================================="
    log " [2/2] Starting Hybrid evaluation"
    log "=========================================="

    stop_server

    start_server "hybrid"
    start_monitor "${RESULTS_DIR}/hybrid_monitor"

    wait_for_server

    log "--- Running Hybrid benchmark ---"
    bash "${SCRIPT_DIR}/benchmark.sh" "hybrid"

    stop_monitor
    stop_server

    log "Hybrid evaluation complete."
}

# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

run_compare() {
    log "=========================================="
    log " Generating comparison report"
    log "=========================================="
    python3 "${SCRIPT_DIR}/compare.py" \
        --results-dir "${RESULTS_DIR}" \
        --gpu-label gpu_only \
        --hybrid-label hybrid
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "Eval starting: MODE=${MODE}"
log "ENV_FILE: ${ENV_FILE}"
log "RUN_TS: ${RUN_TS}"
log "Results path: ${RESULTS_DIR}"
log "------------------------------------------------------------"
log "  Model          : ${MODEL}"
log "  Port           : ${PORT}"
log "  TP size        : ${TENSOR_PARALLEL_SIZE:-1}"
log "  GPU mem util   : ${GPU_MEMORY_UTIL}"
log "  Num prompts    : ${NUM_PROMPTS}"
log "  Input len      : ${INPUT_LEN}"
log "  Output len     : ${OUTPUT_LEN}"
log "  Request rate   : ${REQUEST_RATE}"
log "  CPU engines    : ${HYBRID_NUM_CPU_ENGINES:-1}"
log "  NUMA aware     : ${HYBRID_NUMA_AWARE:-true}"
log "  Routing        : ${HYBRID_ROUTING_STRATEGY:-capacity}"
log "  CPU max seqs   : ${HYBRID_CPU_MAX_SEQS:-0 (auto)}"
log "  CPU kvcache GB : ${HYBRID_CPU_KVCACHE_GB:-0 (auto)}"
log "  CPU threads    : ${HYBRID_CPU_THREADS:-0 (auto)}"
log "  Monitor intv   : ${MONITOR_INTERVAL:-1}s"
log "------------------------------------------------------------"

case "${MODE}" in
    all)
        run_gpu_only
        run_hybrid
        run_compare
        ;;
    gpu_only)
        run_gpu_only
        ;;
    hybrid)
        run_hybrid
        ;;
    compare)
        run_compare
        ;;
    *)
        echo "Usage: $0 <env_file> [all|gpu_only|hybrid|compare]" >&2
        exit 1
        ;;
esac

log "Done."
