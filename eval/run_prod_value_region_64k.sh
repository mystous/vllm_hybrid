#!/usr/bin/env bash
# =============================================================================
# run_prod_value_region_64k.sh — IDE_006 진정한 가치 영역 (≥64K) 측정.
#
# 사용자 지적 (2026-04-28): IDE_006 의 가치 영역은 원래부터 long-context
# (≥32K, KV pool 초과) 였음. 14336 input 측정은 frame 오류였음. 본 회차는
# 64K input × 20 prompts × 8 output 으로 baseline 의 reload-fallback 이
# 한계 도달하는지 측정.
#
# KV size 계산 (Llama-3.3-70B GQA, 80 layers, 320 KB/token):
#   - 20 × 65536 = 1.31M tokens × 320 KB = 420 GB total / 8 worker = 52 GB/worker
#   - worker KV pool ~46 GiB → 명백히 초과 → cold-tier 강제 발화
#
# 비교:
#   - baseline (cold-tier ON, IDE_006 OFF): vLLM native reload-fallback
#   - split_on (cold-tier ON, IDE_006 ON):  IDE_006 partition + (TSK_011 fallback)
#
# 해석:
#   * baseline 이 OOM / hang / 매우 김 → 가치 영역 입증, IDE_006 작업 정당화
#   * baseline 이 충분히 빠름 → 더 큰 영역 (≥128K?) 또는 가치 축 재검토
#
# 실행 시간 견적: 15~30 분 (70B + TP=8 + 1.31M tokens).
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
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_value_region_64k"
mkdir -p "${OUT_DIR}/e2e_artifacts"

MAX_PROMPTS="${MAX_PROMPTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-8}"
LOGPROBS="${LOGPROBS:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

{
    echo "# ${TS}_${HW_TAG}_value_region_64k"
    echo
    echo "- timestamp: ${TS}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo
    echo "## scope — IDE_006 진정한 가치 영역 (≥64K, KV pool 초과)"
    echo "- baseline: ide006_cold_tier_only_value_64k.env  (cold-tier ON, IDE_006 OFF)"
    echo "- split_on: ide006_cold_kv_split_on_value_64k.env (cold-tier ON, IDE_006 ON)"
    echo "- max-prompts: ${MAX_PROMPTS}"
    echo "- max-tokens:  ${MAX_TOKENS}"
    echo "- input-len:   65536  (env file 에서 가져옴)"
    echo "- max-model-len: 65536"
    echo "- KV size estimate: 20 × 65536 × 320KB / 8 worker ≈ 52 GB/worker (KV pool ~46 GiB 초과)"
    echo "- VLLM_COLD_KV_FALLBACK_DEADLINE_MS: ${VLLM_COLD_KV_FALLBACK_DEADLINE_MS}"
} > "${OUT_DIR}/README.md"

log "value region 64k start (max-prompts=${MAX_PROMPTS}, max-tokens=${MAX_TOKENS})"

E2E_RC=0
HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/ide006_cold_tier_only_value_64k.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_value_64k.env" \
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
    echo "## comparison.json"
    echo '```'
    if [[ -f "${OUT_DIR}/e2e_artifacts/comparison.json" ]]; then
        "${PYTHON}" -c "import json; print(json.dumps(json.load(open('${OUT_DIR}/e2e_artifacts/comparison.json')), indent=2, ensure_ascii=False))" 2>&1 | head -30
    else
        echo "(no comparison.json — likely OOM / stall / startup failure)"
    fi
    echo '```'
    echo
    echo "## bench wall time"
    grep -E "batched generate complete:|done in" "${OUT_DIR}/e2e.log" 2>/dev/null || true
    echo
    echo "## firing"
    echo "- TSK_011 cold-fallback fired: $(grep -cE '\[IDE_006/TSK_011 cold-fallback fired' "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)"
    echo "- TSK_004 cold-path fired:    $(grep -cE '\[IDE_006/TSK_004 cold-path fired' "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)"
} >> "${OUT_DIR}/README.md"

log "value 64k done — see ${OUT_DIR}/README.md"
exit 0
