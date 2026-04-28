#!/usr/bin/env bash
# =============================================================================
# run_prod_value_region_128k.sh — IDE_006 극한 가치 영역 (≥128K) 측정.
#
# 64K 회차에서도 baseline 이 충분히 빠름이 확인됨 (50.7s generate). 본 회차
# 는 ≥128K 로 baseline 의 한계 도달 여부 확인:
#   - baseline OOM / hang / 매우 느림 → IDE_006 의 가치 영역 = ≥128K 한정 입증
#   - baseline 충분히 빠름 → IDE_006 가 어떤 영역에서도 baseline 못 이김 → 정의 재검토
#
# KV size: 10 × 131072 × 320 KB / 8 worker ≈ 52 GB/worker (KV pool 46 GiB 1.13×).
#
# 실행 시간 견적: 30분~1시간 (70B + TP=8 + 1.31M tokens, prefill 매우 김).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

export VLLM_COLD_KV_FALLBACK_DEADLINE_MS="${VLLM_COLD_KV_FALLBACK_DEADLINE_MS:-100}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_value_region_128k"
mkdir -p "${OUT_DIR}/e2e_artifacts"

MAX_PROMPTS="${MAX_PROMPTS:-10}"
MAX_TOKENS="${MAX_TOKENS:-8}"
LOGPROBS="${LOGPROBS:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

{
    echo "# ${TS}_${HW_TAG}_value_region_128k"
    echo "- timestamp: ${TS}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo
    echo "## scope — IDE_006 극한 가치 영역 (≥128K)"
    echo "- baseline: ide006_cold_tier_only_value_128k.env  (cold-tier ON, IDE_006 OFF)"
    echo "- split_on: ide006_cold_kv_split_on_value_128k.env (cold-tier ON, IDE_006 ON)"
    echo "- max-prompts: ${MAX_PROMPTS} / max-tokens: ${MAX_TOKENS} / input-len: 131072"
    echo "- KV size estimate: ~52 GB/worker (pool 46 GiB 1.13× 초과)"
} > "${OUT_DIR}/README.md"

log "value 128k start"

E2E_RC=0
HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/ide006_cold_tier_only_value_128k.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_value_128k.env" \
    --max-prompts "${MAX_PROMPTS}" \
    --max-tokens "${MAX_TOKENS}" \
    --logprobs "${LOGPROBS}" \
    --output-dir "${OUT_DIR}/e2e_artifacts" \
    > >(tee "${OUT_DIR}/e2e.log") 2> >(tee "${OUT_DIR}/e2e.stderr.log" >&2) || E2E_RC=$?
log "  e2e exit=${E2E_RC}"

{
    echo
    echo "## exit code: ${E2E_RC}"
    echo
    echo "## comparison.json"
    echo '```'
    if [[ -f "${OUT_DIR}/e2e_artifacts/comparison.json" ]]; then
        "${PYTHON}" -c "import json; print(json.dumps(json.load(open('${OUT_DIR}/e2e_artifacts/comparison.json')), indent=2, ensure_ascii=False))" 2>&1 | head -30
    else
        echo "(no comparison.json — likely OOM / stall)"
    fi
    echo '```'
    echo
    echo "## bench wall time"
    grep -E "batched generate complete:|done in" "${OUT_DIR}/e2e.log" 2>/dev/null || true
    echo
    echo "## firing"
    echo "- TSK_011 cold-fallback fired: $(grep -cE '\[IDE_006/TSK_011 cold-fallback fired' "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)"
    echo "- TSK_004 cold-path fired:    $(grep -cE '\[IDE_006/TSK_004 cold-path fired' "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)"
    echo
    echo "## OOM / 에러 trace"
    grep -niE "out of memory|cuda error|engine_dead|crash" "${OUT_DIR}/e2e.stderr.log" 2>/dev/null | head -5 || echo "(none)"
} >> "${OUT_DIR}/README.md"

log "value 128k done — see ${OUT_DIR}/README.md"
exit 0
