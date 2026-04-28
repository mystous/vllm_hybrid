#!/usr/bin/env bash
# =============================================================================
# run_prod_diag_cold_tier_isolation.sh — TSK_012 사전 진단.
#
# 목적: TSK_011 sweep 에서 발견된 lp ~3.43 발산이 *cold-tier 자체* 의
#   KV source 차이 때문인지, 아니면 *IDE_006 의 attention kernel* (CPU
#   partial-attn + LSE merge / SDPA fallback) 차이 때문인지를 *분리* 측정.
#
# 회차 구성:
#   - baseline: vllm_original_long_ctx.env       (cold-tier 비활성, GPU only)
#   - split_on: ide006_cold_tier_only_long_ctx.env  (cold-tier 활성, IDE_006 비활성)
#   - max-prompts=30, max-tokens=16, logprobs=1  (TSK_011 sweep 와 동일 환경)
#
# 해석:
#   * 두 결과의 worst_max_abs_logprob ≈ TSK_011 sweep 의 ~3.43 → cold-tier
#     자체가 발산 source. TSK_012 (decode reload — cold KV 를 evict 하지 않고
#     계속 GPU 에 두기) 의 design 확정.
#   * 두 결과가 거의 일치 (lp << 1) → cold-tier 자체는 OK. lp ~3.43 발산은
#     IDE_006 의 attention kernel 차이가 원인. TSK_012 의 design 재검토 필요.
#
# 실행 시간 견적: ~12~15 분 (70B + TP=8, baseline + split_on 각 회차).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

# libcuda.so.1 search path workaround.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"
# 외부망 차단.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_diag_cold_tier_iso"
mkdir -p "${OUT_DIR}/e2e_artifacts"

MAX_PROMPTS="${MAX_PROMPTS:-30}"
MAX_TOKENS="${MAX_TOKENS:-16}"
LOGPROBS="${LOGPROBS:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

{
    echo "# ${TS}_${HW_TAG}_diag_cold_tier_iso"
    echo
    echo "- timestamp: ${TS}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo
    echo "## scope"
    echo "- baseline: vllm_original_long_ctx.env       (cold-tier 비활성)"
    echo "- split_on: ide006_cold_tier_only_long_ctx.env  (cold-tier 활성, IDE_006 비활성)"
    echo "- max-prompts: ${MAX_PROMPTS}"
    echo "- max-tokens:  ${MAX_TOKENS}"
    echo "- logprobs:    ${LOGPROBS}"
    echo
    echo "## 진단 결정 분기"
    echo "- worst_lp ≈ 3.43 → cold-tier 자체가 발산 source → TSK_012 진행"
    echo "- worst_lp ≪ 1   → cold-tier 는 OK → TSK_012 design 재검토"
} > "${OUT_DIR}/README.md"

log "diag start (cold-tier isolation, max-prompts=${MAX_PROMPTS})"

E2E_RC=0
HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/vllm_original_long_ctx.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_tier_only_long_ctx.env" \
    --max-prompts "${MAX_PROMPTS}" \
    --max-tokens "${MAX_TOKENS}" \
    --logprobs "${LOGPROBS}" \
    --output-dir "${OUT_DIR}/e2e_artifacts" \
    --allow-equivalent-config \
    > >(tee "${OUT_DIR}/e2e.log") 2> >(tee "${OUT_DIR}/e2e.stderr.log" >&2) || E2E_RC=$?
log "  e2e exit=${E2E_RC}"

{
    echo
    echo "## exit code"
    echo "- e2e RC: ${E2E_RC}"
    echo
    echo "## comparison.json (jq)"
    echo '```'
    "${PYTHON}" -c "import json; print(json.dumps(json.load(open('${OUT_DIR}/e2e_artifacts/comparison.json')), indent=2, ensure_ascii=False))" 2>&1 | head -30
    echo '```'
} >> "${OUT_DIR}/README.md"

# 진단 자체는 e2e RC 에 무관 — verdict 가 FAIL 이어도 측정값을 가지고 결정.
log "diag complete — see ${OUT_DIR}/e2e_artifacts/comparison.json"
exit 0
