#!/usr/bin/env bash
# =============================================================================
# run_prod_quick_tst003.sh — TST_003 e2e accuracy 짧은 회차.
#
# 목적: dev smoke 에서 발견된 prompt2 발산 (lp=1.10, ppl_rel=0.099) 이
#   prod 머신에서도 재현되는지 D-ii (per-position logprob diff + 시퀀스
#   PPL relative diff) 정량으로 확인.
#
# 회차 구성:
#   - baseline:  vllm_original_long_ctx.env (Llama-3.3-70B + TP=8, split off)
#   - split_on:  ide006_cold_kv_split_on_long_ctx.env (split on)
#   - max-prompts=30 (prompt2 포함 보장 + KV pool overflow 로 cold 발화)
#   - max-tokens=16
#   - logprobs=1   (D-ii 측정 필수. cold_verify 회차의 logprobs=0 와 다름)
#
# 실행 시간 견적: ~10~15 분
#   (baseline 5~7 분 + split_on 5~7 분, 각각 70B+TP=8 startup 포함)
#
# Usage:
#   bash eval/run_prod_quick_tst003.sh
#
# 산출물: eval/results/<TS>_<HW_TAG>_quick_tst003/
#   ├── README.md
#   ├── e2e.log / e2e.stderr.log
#   └── e2e_artifacts/
#       ├── baseline.json     (D-ii 의 reference)
#       ├── split_on.json     (D-ii 의 비교 대상)
#       └── comparison.json   (run_e2e_accuracy.py 의 자동 비교 산출)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

# libcuda.so.1 search path workaround on this prod box.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"
# 외부망 차단 (방화벽 환경).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_quick_tst003"
mkdir -p "${OUT_DIR}/e2e_artifacts"

MAX_PROMPTS="${MAX_PROMPTS:-30}"
MAX_TOKENS="${MAX_TOKENS:-16}"
LOGPROBS="${LOGPROBS:-1}"

# TSK_011 §4.1/4.2 — deadline-aware cold path + GPU full FA fallback.
# 100 ms default (layer hot FA 시간의 ~4×). 0 이면 비활성.
export VLLM_COLD_KV_FALLBACK_DEADLINE_MS="${VLLM_COLD_KV_FALLBACK_DEADLINE_MS:-100}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

{
    echo "# ${TS}_${HW_TAG}_quick_tst003"
    echo
    echo "- timestamp: ${TS}"
    echo "- hw_tag:    ${HW_TAG}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo
    echo "## scope"
    echo "- baseline: vllm_original_long_ctx.env (Llama-3.3-70B + TP=8, split off)"
    echo "- split_on: ide006_cold_kv_split_on_long_ctx.env"
    echo "- max-prompts: ${MAX_PROMPTS}"
    echo "- max-tokens:  ${MAX_TOKENS}"
    echo "- logprobs:    ${LOGPROBS}  (D-ii 측정 필수)"
    echo
    echo "## 통과 기준"
    echo "- e2e RC = 0"
    echo "- comparison.json 의 D-ii (logprob max abs / PPL rel) 가 tolerance 내"
    echo "- prompt2 의 발산 여부 정량 확인"
} > "${OUT_DIR}/README.md"

log "starting TST_003 quick (max-prompts=${MAX_PROMPTS}, logprobs=${LOGPROBS})"

E2E_RC=0
HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/vllm_original_long_ctx.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_long_ctx.env" \
    --max-prompts "${MAX_PROMPTS}" \
    --max-tokens "${MAX_TOKENS}" \
    --logprobs "${LOGPROBS}" \
    --output-dir "${OUT_DIR}/e2e_artifacts" \
    > >(tee "${OUT_DIR}/e2e.log") 2> >(tee "${OUT_DIR}/e2e.stderr.log" >&2) || E2E_RC=$?
log "  e2e exit=${E2E_RC}"

{
    echo
    echo "## exit code"
    echo "- e2e RC: ${E2E_RC}"
    echo
    echo "## comparison.json (jq)"
    echo '```'
    "${PYTHON}" -c "import json; print(json.dumps(json.load(open('${OUT_DIR}/e2e_artifacts/comparison.json')), indent=2, ensure_ascii=False))" 2>&1 | head -60
    echo '```'
} >> "${OUT_DIR}/README.md"

if [[ ${E2E_RC} -eq 0 ]]; then
    log "PASS — TST_003 quick OK"
else
    log "FAIL — see ${OUT_DIR}/e2e.stderr.log"
fi

exit ${E2E_RC}
