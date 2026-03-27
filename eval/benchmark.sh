#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — vLLM 벤치마크 실행 스크립트
# 사용법:
#   ./benchmark.sh <label>
#   label: gpu_only | hybrid  (결과 파일명 prefix로 사용)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] .env 파일을 찾을 수 없습니다: $ENV_FILE" >&2
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
echo " 벤치마크 시작: LABEL=${LABEL}"
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

echo "[benchmark.sh] 완료: ${RESULT_FILE}"
