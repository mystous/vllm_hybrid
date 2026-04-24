#!/usr/bin/env bash
# =============================================================================
# serve.sh — Start vLLM OpenAI-compatible server
#
# Usage:
#   ./serve.sh [env_file]
#
# Examples:
#   ./serve.sh                         # uses envs/default.env
#   ./serve.sh envs/dev_rtx3090.env
#
# Tip: run in foreground to watch logs, or via run.sh for full pipeline.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/envs/default.env}"

if [[ ! "${ENV_FILE}" = /* ]]; then
    ENV_FILE="${SCRIPT_DIR}/${ENV_FILE}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[serve.sh] env file not found: ${ENV_FILE}" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${MODEL:?MODEL is not set}"
: "${PORT:=8000}"
: "${HOST:=127.0.0.1}"
: "${GPU_MEMORY_UTIL:=0.9}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${MAX_MODEL_LEN:=4096}"
: "${EXTRA_SERVE_ARGS:=}"

echo "============================================================"
echo " vLLM serve"
echo "   MODEL = ${MODEL}"
echo "   HOST:PORT = ${HOST}:${PORT}"
echo "   TP = ${TENSOR_PARALLEL_SIZE}, GPU_MEM_UTIL = ${GPU_MEMORY_UTIL}"
echo "   MAX_MODEL_LEN = ${MAX_MODEL_LEN}"
echo "   ENV_FILE = ${ENV_FILE}"
echo "============================================================"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"

exec "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${EXTRA_SERVE_ARGS}
