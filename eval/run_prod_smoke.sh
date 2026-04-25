#!/usr/bin/env bash
# =============================================================================
# run_prod_smoke.sh — prod 서버에서 다음을 한 번에 실행하고 결과를 저장한다:
#   - TST_001 (TSK_001 dev kernel: A · B(i) · C) — 87 case 재현
#   - TST_004 (TSK_003 prod SIMD cross-check: B(ii) AVX-512 + B(iii) AMX) —
#       TSK_003 §4.2a/§4.2b kernel 빌드 + wrapper enable 후 자동 활성. pytest
#       collection 이 `tests/v1/cpu_partial_attention/test_avx512_cross_check.py`
#       와 `test_amx_cross_check.py` 를 자동 포함. 미빌드 시에는 skipif 마커로
#       skip 되어 0 fail.
#   - IDE_006 long-context e2e 시나리오 (vllm_original / ide006_cold_kv 두 env)
#
# Usage:
#   bash eval/run_prod_smoke.sh             # 실행 + 결과 저장 (push 는 직접)
#   bash eval/run_prod_smoke.sh --push      # 실행 + commit + push (branch 그대로)
#
# 출력 위치:
#   eval/results/prod_smoke_<TS>_<HW_TAG>/
#     ├── pytest.log              # pytest stdout (TST_001 + TST_004)
#     ├── pytest_junit.xml        # pytest JUnit (CI/툴 호환)
#     ├── isa_info.txt            # /proc/cpuinfo + nvidia-smi snapshot
#     └── README.md               # 메타 (git rev, vllm version, 실행 환경)
#   eval/results/<TS>_<HW_TAG>_<MODEL>/        ← run.sh 산출물 (server.log, bench.json,
#   eval/results/<TS+1>_<HW_TAG>_<MODEL>/        monitor_*.csv 등) — env 별 1 개씩
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

TS="$(date '+%Y%m%d_%H%M%S')"
SMOKE_DIR="${SCRIPT_DIR}/results/prod_smoke_${TS}_${HW_TAG}"
mkdir -p "${SMOKE_DIR}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python)"
fi

PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --------------------------------------------------------------------- meta

log "writing meta to ${SMOKE_DIR}/README.md"
{
    echo "# prod_smoke_${TS}_${HW_TAG}"
    echo
    echo "- timestamp: ${TS}"
    echo "- hw_tag:    ${HW_TAG}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo
    echo "## components"
    echo "- pytest TST_001 (TSK_001 dev kernel: 단계 A · B(i) · C — dev 에서 통과한 87 testcase 재현)"
    echo "- pytest TST_004 (TSK_003 prod SIMD: B(ii) portable vs AVX-512 + B(iii) portable vs AMX)"
    echo "    └─ TSK_003 §4.2a/§4.2b kernel 미빌드 시 skipif 마커로 skip"
    echo "- eval/run.sh envs/vllm_original_long_ctx.env (split-off long-context baseline)"
    echo "- eval/run.sh envs/ide006_cold_kv_long_ctx.env (cold-tier KV offload)"
    echo
    echo "Result subdirs (run.sh): see eval/results/<TS>_<HW_TAG>_<MODEL>/"
} > "${SMOKE_DIR}/README.md"

log "capturing CPU/GPU snapshot to ${SMOKE_DIR}/isa_info.txt"
{
    echo "=== /proc/cpuinfo (ISA flags) ==="
    cat /proc/cpuinfo | grep -oE 'avx512[a-z_0-9]*|amx_[a-z_]*|avx2|sse4_2' | sort -u
    echo
    echo "=== nvidia-smi ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>&1 || echo "nvidia-smi unavailable"
    echo
    echo "=== compiler ==="
    c++ --version 2>&1 | head -1
} > "${SMOKE_DIR}/isa_info.txt"

# --------------------------------------------------------------------- 1) pytest

log "[1/3] pytest TST_001 (TSK_001 dev kernel) + TST_004 (TSK_003 prod SIMD, skip if unbuilt)"
PYTEST_RC=0
"${PYTHON}" -m pytest tests/v1/cpu_partial_attention/ -v \
    --junit-xml="${SMOKE_DIR}/pytest_junit.xml" \
    2>&1 | tee "${SMOKE_DIR}/pytest.log" || PYTEST_RC=$?

# --------------------------------------------------------------------- 2) baseline scenario

log "[2/3] scenario: vllm_original_long_ctx (split-off baseline)"
SCEN1_RC=0
bash "${SCRIPT_DIR}/run.sh" envs/vllm_original_long_ctx.env || SCEN1_RC=$?
log "  scenario 1 exit=${SCEN1_RC}"

# --------------------------------------------------------------------- 3) cold-tier scenario

log "[3/3] scenario: ide006_cold_kv_long_ctx (OffloadingConnector)"
SCEN2_RC=0
bash "${SCRIPT_DIR}/run.sh" envs/ide006_cold_kv_long_ctx.env || SCEN2_RC=$?
log "  scenario 2 exit=${SCEN2_RC}"

# --------------------------------------------------------------------- summary

{
    echo
    echo "## exit codes"
    echo "- pytest:               ${PYTEST_RC}"
    echo "- scenario baseline:    ${SCEN1_RC}"
    echo "- scenario cold_kv:     ${SCEN2_RC}"
} >> "${SMOKE_DIR}/README.md"

log "smoke artifacts → ${SMOKE_DIR}"
log "scenario artifacts → eval/results/${TS}_${HW_TAG}_*/ (run.sh 자동 생성)"

# --------------------------------------------------------------------- push (optional)

if [[ ${PUSH} -eq 1 ]]; then
    log "staging eval/results/ + git push"
    git add eval/results/
    if git diff --cached --quiet; then
        log "  nothing to commit"
    else
        git commit -m "chore(eval): prod smoke run ${TS} @ ${HW_TAG}"
        BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        git push origin "${BRANCH}"
    fi
else
    cat <<EOF

[push 안내] 결과를 push 하려면:
  git add eval/results/
  git commit -m "chore(eval): prod smoke run ${TS} @ ${HW_TAG}"
  git push origin $(git rev-parse --abbrev-ref HEAD)

또는 다음 실행 시 \`bash eval/run_prod_smoke.sh --push\` 로 자동.
EOF
fi

# overall exit
[[ ${PYTEST_RC} -eq 0 && ${SCEN1_RC} -eq 0 && ${SCEN2_RC} -eq 0 ]] && exit 0 || exit 1
