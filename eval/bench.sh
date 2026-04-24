#!/usr/bin/env bash
# =============================================================================
# bench.sh — Run vLLM benchmark against an already-running server
#
# Usage:
#   ./bench.sh [env_file] [result_dir]
#
# Args (all optional):
#   env_file    path to .env (default: envs/default.env)
#   result_dir  output directory (default: results/<timestamp>_<model>)
#
# Examples:
#   ./bench.sh
#   ./bench.sh envs/default.env
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/envs/default.env}"
RESULT_DIR_ARG="${2:-}"

if [[ ! "${ENV_FILE}" = /* ]]; then
    ENV_FILE="${SCRIPT_DIR}/${ENV_FILE}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[bench.sh] env file not found: ${ENV_FILE}" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${MODEL:?MODEL is not set}"
: "${PORT:=8000}"
: "${HOST:=127.0.0.1}"
: "${BACKEND:=vllm}"
: "${DATASET_NAME:=random}"
: "${NUM_PROMPTS:=200}"
: "${INPUT_LEN:=512}"
: "${OUTPUT_LEN:=256}"
: "${REQUEST_RATE:=inf}"
: "${RESULTS_DIR:=results}"

MODEL_SHORT="${MODEL##*/}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

if [[ -n "${RESULT_DIR_ARG}" ]]; then
    RUN_DIR="${RESULT_DIR_ARG}"
else
    TS="$(date '+%Y%m%d_%H%M%S')"
    RUN_DIR="${SCRIPT_DIR}/${RESULTS_DIR}/${TS}_${HW_TAG}_${MODEL_SHORT}"
fi
mkdir -p "${RUN_DIR}"

RESULT_FILE="${RUN_DIR}/bench.json"
LOG_FILE="${RUN_DIR}/bench.log"
SYSINFO_FILE="${RUN_DIR}/system_info.json"

echo "============================================================"
echo " vLLM bench"
echo "   MODEL = ${MODEL}"
echo "   TARGET = http://${HOST}:${PORT}"
echo "   DATASET = ${DATASET_NAME} (in=${INPUT_LEN}, out=${OUTPUT_LEN})"
echo "   PROMPTS = ${NUM_PROMPTS}, RATE = ${REQUEST_RATE}"
echo "   RESULT → ${RESULT_FILE}"
echo "============================================================"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"

# Collect host/CPU/GPU/software info
"${PYTHON}" "${SCRIPT_DIR}/sysinfo.py" "${SYSINFO_FILE}" || true

"${PYTHON}" -m vllm.entrypoints.cli.main bench serve \
    --backend "${BACKEND}" \
    --base-url "http://${HOST}:${PORT}" \
    --model "${MODEL}" \
    --dataset-name "${DATASET_NAME}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --save-result \
    --result-filename "${RESULT_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo "[bench.sh] Done → ${RESULT_FILE}"
