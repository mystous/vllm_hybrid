#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — vLLM benchmark execution script
# Usage:
#   ./benchmark.sh <label>
#   label: gpu_only | hybrid  (used as result filename prefix)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] .env file not found: $ENV_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

LABEL="${1:-benchmark}"
# EVAL_RUN_DIR is set by run_eval.sh (timestamped subdir).
# When running benchmark.sh standalone, fall back to .env RESULTS_DIR.
if [[ -n "${EVAL_RUN_DIR:-}" ]]; then
    RESULTS_DIR="${EVAL_RUN_DIR}"
else
    RUN_TS="$(date '+%Y%m%d_%H%M%S')"
    RESULTS_DIR="${SCRIPT_DIR}/${RESULTS_DIR:-results}/${RUN_TS}"
fi
mkdir -p "$RESULTS_DIR"

RESULT_FILE="${RESULTS_DIR}/${LABEL}.json"
LOG_FILE="${RESULTS_DIR}/${LABEL}_bench.log"

echo "============================================================"
echo " Benchmark starting: LABEL=${LABEL}"
echo " MODEL=${MODEL}, NUM_PROMPTS=${NUM_PROMPTS}"
echo " INPUT_LEN=${INPUT_LEN}, OUTPUT_LEN=${OUTPUT_LEN}"
echo " RESULT → ${RESULT_FILE}"
echo "============================================================"

python "${VLLM_ROOT}/benchmarks/benchmark_serving.py" \
    --backend vllm \
    --base-url "http://localhost:${PORT}" \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --save-result \
    --result-filename "${RESULT_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo "[benchmark.sh] Done: ${RESULT_FILE}"
