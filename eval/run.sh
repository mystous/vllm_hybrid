#!/usr/bin/env bash
# =============================================================================
# run.sh — Full pipeline: start server → monitor → bench → shutdown
#
# Usage:
#   ./run.sh [env_file]
#
# Examples:
#   ./run.sh
#   ./run.sh envs/default.env
#
# Outputs (under results/<timestamp>_<model>/):
#   server.log            vLLM server stdout/stderr
#   bench.json            benchmark result
#   bench.log             benchmark stdout
#   monitor_cpu.csv       CPU utilization samples
#   monitor_gpu.csv       GPU utilization samples
#   monitor.log           monitor stdout
#   env.snapshot          copy of env file used
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/envs/default.env}"

if [[ ! "${ENV_FILE}" = /* ]]; then
    ENV_FILE="${SCRIPT_DIR}/${ENV_FILE}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[run.sh] env file not found: ${ENV_FILE}" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${MODEL:?MODEL is not set}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${RESULTS_DIR:=results}"
: "${MONITOR_INTERVAL:=1}"
: "${SERVER_READY_TIMEOUT:=300}"
: "${SERVER_READY_POLL:=3}"

MODEL_SHORT="${MODEL##*/}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

TS="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${SCRIPT_DIR}/${RESULTS_DIR}/${TS}_${HW_TAG}_${MODEL_SHORT}"
mkdir -p "${RUN_DIR}"
cp "${ENV_FILE}" "${RUN_DIR}/env.snapshot"

SERVER_LOG="${RUN_DIR}/server.log"
MONITOR_LOG="${RUN_DIR}/monitor.log"
MONITOR_PREFIX="${RUN_DIR}/monitor"

SERVER_PID=""
MONITOR_PID=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup() {
    log "cleanup…"
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "stopping server (PID=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        # vLLM spawns workers; wait then force-cleanup
        sleep 2
        pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local url="http://${HOST}:${PORT}/health"
    local elapsed=0
    log "waiting for server at ${url} (timeout=${SERVER_READY_TIMEOUT}s)"
    while ! curl -sf "${url}" >/dev/null 2>&1; do
        if [[ ${elapsed} -ge ${SERVER_READY_TIMEOUT} ]]; then
            log "ERROR: server not ready after ${SERVER_READY_TIMEOUT}s"
            return 1
        fi
        sleep "${SERVER_READY_POLL}"
        elapsed=$((elapsed + SERVER_READY_POLL))
        log "  waiting… ${elapsed}/${SERVER_READY_TIMEOUT}s"
    done
    log "server ready (after ${elapsed}s)"
}

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"

log "run dir: ${RUN_DIR}"

# 1) Start server
log "starting server → ${SERVER_LOG}"
bash "${SCRIPT_DIR}/serve.sh" "${ENV_FILE}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
log "server PID: ${SERVER_PID}"

# 2) Start monitor (records even during warmup)
log "starting monitor (interval=${MONITOR_INTERVAL}s)"
"${PYTHON}" "${SCRIPT_DIR}/monitor.py" "${MONITOR_PREFIX}" \
    --interval "${MONITOR_INTERVAL}" >"${MONITOR_LOG}" 2>&1 &
MONITOR_PID=$!
log "monitor PID: ${MONITOR_PID}"

# 3) Wait for server readiness
wait_for_server

# 4) Run benchmark
log "running benchmark"
bash "${SCRIPT_DIR}/bench.sh" "${ENV_FILE}" "${RUN_DIR}"

# cleanup trap handles shutdown
log "done. results → ${RUN_DIR}"
