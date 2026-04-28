#!/usr/bin/env bash
# =============================================================================
# run_prod_quick_tst003_value_region.sh — IDE_006 가치 영역 (long-context,
#   KV pool 초과) 에서의 *순 increment* 측정.
#
# 배경: 이전 prod TST_003 회차들 (max-prompts=30) 은 KV pool 한계 안에 들어가
#   cold-tier 가 dead weight 였음. 게다가 baseline=vllm_original_long_ctx
#   (GPU only) 로 NUM_PROMPTS=100 시도 시 OOM / chunked prefill stall 로 비교
#   불가능. 본 wrapper 는 다음으로 해결:
#
#   1. baseline 을 cold-tier-only env (cold-tier ON, IDE_006 OFF) 로 변경 →
#      양쪽 모두 PCIe 비용 동등화 → wall-time 차이가 IDE_006 의 *순 increment*
#      만 반영. 또한 OOM 회피.
#   2. OUTPUT_LEN=8 (decode 짧게) — KV pool 초과는 prefill 단계에서 보장됨.
#   3. NUM_PROMPTS=100 (default) — KV pool (~46 GiB worker per) 초과 영역 보장:
#      100 prompts × 14336 × 320 KB / 8 worker ≈ 57 GB worker per > 46 GB.
#
# IDE_006 §10 — Long-context 영역에서만 cold KV 자연 발생. 본 회차가 그 영역.
#
# Usage:
#   bash eval/run_prod_quick_tst003_value_region.sh
#   VLLM_COLD_KV_FALLBACK_DEADLINE_MS=1000 bash eval/run_prod_quick_tst003_value_region.sh
#
# 실행 시간 견적: ~15~20 분 (baseline + split_on, 70B + TP=8).
#
# 산출물: eval/results/<TS>_<HW_TAG>_quick_tst003_value/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

# libcuda.so.1 search path workaround on this prod box.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"
# 외부망 차단.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# TSK_011 fallback default 100 ms (변경 가능). 0 이면 비활성.
export VLLM_COLD_KV_FALLBACK_DEADLINE_MS="${VLLM_COLD_KV_FALLBACK_DEADLINE_MS:-100}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_quick_tst003_value"
mkdir -p "${OUT_DIR}/e2e_artifacts"

MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_TOKENS="${MAX_TOKENS:-8}"
LOGPROBS="${LOGPROBS:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

{
    echo "# ${TS}_${HW_TAG}_quick_tst003_value"
    echo
    echo "- timestamp: ${TS}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo
    echo "## scope — IDE_006 가치 영역 (long-context, KV pool 초과)"
    echo "- baseline: ide006_cold_tier_only_long_ctx.env  (cold-tier ON, IDE_006 OFF — PCIe 비용 동등화)"
    echo "- split_on: ide006_cold_kv_split_on_long_ctx.env (cold-tier ON, IDE_006 ON)"
    echo "- max-prompts: ${MAX_PROMPTS}  (KV pool ~46 GiB worker per 초과 — cold-tier 강제 발화)"
    echo "- max-tokens:  ${MAX_TOKENS}   (decode 단축, KV pool 초과는 prefill 에서 보장)"
    echo "- logprobs:    ${LOGPROBS}"
    echo "- VLLM_COLD_KV_FALLBACK_DEADLINE_MS: ${VLLM_COLD_KV_FALLBACK_DEADLINE_MS}"
    echo
    echo "## 의도"
    echo "- 두 회차 모두 PCIe 비용 동등화 → wall-time 차이가 IDE_006 *순 increment*"
    echo "- 두 회차 모두 cold-tier 동일 source → lp 차이가 IDE_006 attention kernel *순 발산*"
} > "${OUT_DIR}/README.md"

log "value region run start (max-prompts=${MAX_PROMPTS}, max-tokens=${MAX_TOKENS})"

E2E_RC=0
HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/ide006_cold_tier_only_long_ctx.env" \
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
    "${PYTHON}" -c "import json; print(json.dumps(json.load(open('${OUT_DIR}/e2e_artifacts/comparison.json')), indent=2, ensure_ascii=False))" 2>&1 | head -30
    echo '```'
    echo
    echo "## bench wall time"
    grep -E "batched generate complete:|done in" "${OUT_DIR}/e2e.log" 2>/dev/null || true
    echo
    echo "## fallback firing"
    echo "- TSK_011 cold-fallback 발동 횟수: $(grep -cE '\[IDE_006/TSK_011 cold-fallback fired' "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)"
    echo "- TSK_004 cold-path 발동 횟수:    $(grep -cE '\[IDE_006/TSK_004 cold-path fired' "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)"
} >> "${OUT_DIR}/README.md"

log "value region run done — see ${OUT_DIR}/README.md"
exit 0
