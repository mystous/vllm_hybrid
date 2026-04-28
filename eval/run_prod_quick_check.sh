#!/usr/bin/env bash
# =============================================================================
# run_prod_quick_check.sh — prod 머신에서 IDE_006 *현재 구현* 만 빠르게 인증.
#
# 범위: contract 단위 회귀 (pytest) only. e2e / vllm engine 띄우는 영역은
#       의도적으로 제외 ("오래 진행 안 함" 요구).
#
# 검증 대상:
#   - tests/v1/cpu_partial_attention/  (TSK_001 kernel + TSK_002 §4.5b/§4.5c/§4.6
#       fail-closed gate / mask helper / packing helper / scatter buf cache /
#       merge event race / wrapper dispatch / SIMD cross-check 등)
#   - tests/v1/kv_offload/             (TSK_004 NUMA partition + reload sync)
#
# 실행 시간 견적: ~5~10 분 (415 cases collected).
#
# Usage:
#   bash eval/run_prod_quick_check.sh
#
# 산출물: eval/results/<TS>_<HW_TAG>_quick_check/
#   ├── README.md           meta (commit / vllm rev / scope)
#   ├── pytest.log          pytest stdout
#   └── pytest_junit.xml    JUnit XML
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

# libcuda.so.1 search path workaround on this prod box.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"

# 본 환경은 외부망 차단 (방화벽). HF Hub 호출 자체를 막아 gated/offline race
# 회귀를 사전 봉쇄. 필요한 모델은 사전 수동 배포된 /root/.cache/huggingface
# 에서만 사용.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"

TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_quick_check"
mkdir -p "${OUT_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- meta ----------------------------------------------------------------
{
    echo "# ${TS}_${HW_TAG}_quick_check"
    echo
    echo "- timestamp: ${TS}"
    echo "- hw_tag:    ${HW_TAG}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo
    echo "## scope"
    echo "- tests/v1/cpu_partial_attention/  (TSK_001 + TSK_002 회귀)"
    echo "- tests/v1/kv_offload/             (TSK_004 NUMA + reload sync)"
    echo "- 제외: tests/v1/kv_offload/test_cpu_offloading.py (vLLM upstream"
    echo "  inherited 테스트 — IDE_006 본 회귀 아님. Llama-3.2-1B 의존)"
    echo "- e2e 미포함 — '오래 진행하지 말고 개발된 내용 확인만' (사용자 요구)"
    echo "- 모드: offline (방화벽 환경, HF_HUB_OFFLINE=1)"
} > "${OUT_DIR}/README.md"

log "scope: contract regression only (pytest)"
log "output: ${OUT_DIR}"

# --- pytest --------------------------------------------------------------
set +e
"${PYTHON}" -m pytest \
    tests/v1/cpu_partial_attention/ \
    tests/v1/kv_offload/ \
    --deselect tests/v1/kv_offload/test_cpu_offloading.py \
    -v \
    --tb=short \
    --junit-xml="${OUT_DIR}/pytest_junit.xml" \
    2>&1 | tee "${OUT_DIR}/pytest.log"
PYTEST_RC=${PIPESTATUS[0]}
set -e

# --- summary -------------------------------------------------------------
{
    echo
    echo "## result"
    echo "- pytest_rc: ${PYTEST_RC}"
    echo
    echo "### tail"
    echo '```'
    tail -15 "${OUT_DIR}/pytest.log"
    echo '```'
} >> "${OUT_DIR}/README.md"

if [[ ${PYTEST_RC} -eq 0 ]]; then
    log "PASS — quick_check OK"
else
    log "FAIL — see ${OUT_DIR}/pytest.log"
fi

exit ${PYTEST_RC}
