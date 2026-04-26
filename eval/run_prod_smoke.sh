#!/usr/bin/env bash
# =============================================================================
# run_prod_smoke.sh — on a prod server, run the following in one shot and
#   save results:
#   - TST_001 (TSK_001 dev kernel: A · B(i) · C) — reproduces 87 cases.
#   - TST_004 (TSK_003 prod SIMD cross-check: B(ii) AVX-512 + B(iii) AMX) —
#       activates automatically once the TSK_003 §4.2a/§4.2b kernels are
#       built and the wrapper is enabled. pytest collection automatically
#       picks up `tests/v1/cpu_partial_attention/test_avx512_cross_check.py`
#       and `test_amx_cross_check.py`. While the kernels are unbuilt the
#       skipif marker keeps the suite at 0 fail.
#   - IDE_006 long-context e2e scenarios (vllm_original / ide006_cold_kv /
#     ide006_cold_kv_split_on envs — all on meta-llama/Llama-3.3-70B-Instruct
#     with TP=8).
#   - TST_003 e2e accuracy gate (run_e2e_accuracy.py) — D-i token divergence
#     + D-ii logprob / PPL diff between baseline and Phase 4c split_on. Same
#     model + TP, just enough prompts and max_tokens to surface algorithmic
#     correctness without dominating the smoke wall time.
#
# Usage:
#   bash eval/run_prod_smoke.sh             # run + save (push manually)
#   bash eval/run_prod_smoke.sh --push      # run + commit + push (current branch)
#
# Output layout (follows the eval/results/ convention <TS>_<HW_TAG>_<TAG>/):
#   eval/results/<TS>_<HW_TAG>_prod_smoke/    <- this script's own artifacts
#     ├── pytest.log              # pytest stdout (TST_001 + TST_004)
#     ├── pytest_junit.xml        # pytest JUnit (CI/tooling compatible)
#     ├── isa_info.txt            # /proc/cpuinfo + nvidia-smi snapshot
#     └── README.md               # meta (git rev, vllm version, exec env)
#   eval/results/<TS+1>_<HW_TAG>_<MODEL>/     <- child run.sh artifacts (server.log,
#   eval/results/<TS+2>_<HW_TAG>_<MODEL>/         bench.json, monitor_*.csv) —
#                                                 one per env. Adjacent in `ls -t`
#                                                 because the parent shares the
#                                                 same TS prefix.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

TS="$(date '+%Y%m%d_%H%M%S')"
SMOKE_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_prod_smoke"
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
    echo "# ${TS}_${HW_TAG}_prod_smoke"
    echo
    echo "- timestamp: ${TS}"
    echo "- hw_tag:    ${HW_TAG}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo
    echo "## components"
    echo "- pytest TST_001 (TSK_001 dev kernel: stages A, B(i), C — reproduces 87 dev testcases)"
    echo "- pytest TST_004 (TSK_003 prod SIMD: B(ii) portable vs AVX-512 + B(iii) portable vs AMX)"
    echo "    └─ skipped via skipif marker if the TSK_003 §4.2a/§4.2b kernels are not built"
    echo "- eval/run.sh envs/vllm_original_long_ctx.env (split-off baseline, Llama-3.3-70B + TP=8)"
    echo "- eval/run.sh envs/ide006_cold_kv_long_ctx.env (OffloadingConnector only)"
    echo "- eval/run.sh envs/ide006_cold_kv_split_on_long_ctx.env (full Cold-KV CPU partial attention — TSK_002 Phase 4c)"
    echo "- eval/run_e2e_accuracy.py (TST_003 D-i token divergence + D-ii logprob/PPL — TSK_002 검증 게이트)"
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

log "[1/5] pytest TST_001 (TSK_001 dev kernel) + TST_004 (TSK_003 prod SIMD, skip if unbuilt)"
PYTEST_RC=0
"${PYTHON}" -m pytest tests/v1/cpu_partial_attention/ -v \
    --junit-xml="${SMOKE_DIR}/pytest_junit.xml" \
    2>&1 | tee "${SMOKE_DIR}/pytest.log" || PYTEST_RC=$?

# --------------------------------------------------------------------- 2) baseline scenario

log "[2/5] scenario: vllm_original_long_ctx (split-off baseline, Llama-3.3-70B + TP=8)"
SCEN1_RC=0
bash "${SCRIPT_DIR}/run.sh" envs/vllm_original_long_ctx.env || SCEN1_RC=$?
log "  scenario 1 exit=${SCEN1_RC}"

# --------------------------------------------------------------------- 3) cold-tier scenario

log "[3/5] scenario: ide006_cold_kv_long_ctx (OffloadingConnector only)"
SCEN2_RC=0
bash "${SCRIPT_DIR}/run.sh" envs/ide006_cold_kv_long_ctx.env || SCEN2_RC=$?
log "  scenario 2 exit=${SCEN2_RC}"

# --------------------------------------------------------------------- 4) cold-tier + split-on (TSK_002 Phase 4c)

log "[4/5] scenario: ide006_cold_kv_split_on_long_ctx (Cold-KV CPU partial attention — TSK_002 §4.5 Phase 4c)"
SCEN3_RC=0
bash "${SCRIPT_DIR}/run.sh" envs/ide006_cold_kv_split_on_long_ctx.env || SCEN3_RC=$?
log "  scenario 3 exit=${SCEN3_RC}"

# --------------------------------------------------------------------- 5) e2e accuracy (TST_003 D-i + D-ii)

log "[5/5] e2e accuracy (TST_003: D-i token divergence + D-ii logprob/PPL)"
ACC_RC=0
ACC_OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_e2e_accuracy"
mkdir -p "${ACC_OUT_DIR}"
# baseline / split_on env 는 [2] / [4] 에서 사용한 동일한 env 파일을 그대로
# 재사용. 모델·TP·max_model_len·EXTRA_SERVE_ARGS 가 단일 source of truth.
HW_TAG="${HW_TAG}" "${PYTHON}" "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/vllm_original_long_ctx.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_long_ctx.env" \
    --max-tokens 64 \
    --logprobs 20 \
    --output-dir "${ACC_OUT_DIR}" \
    2>&1 | tee "${ACC_OUT_DIR}/run_e2e_accuracy.log" || ACC_RC=$?
log "  e2e accuracy exit=${ACC_RC}"

# --------------------------------------------------------------------- summary

{
    echo
    echo "## exit codes"
    echo "- pytest:                       ${PYTEST_RC}"
    echo "- scenario baseline:            ${SCEN1_RC}"
    echo "- scenario cold_kv (offload):   ${SCEN2_RC}"
    echo "- scenario cold_kv split-on:    ${SCEN3_RC}"
    echo "- e2e accuracy (TST_003):       ${ACC_RC}"
} >> "${SMOKE_DIR}/README.md"

log "smoke artifacts -> ${SMOKE_DIR}"
log "scenario artifacts -> eval/results/${TS}_${HW_TAG}_*/ (auto-created by run.sh)"

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

[push hint] To push the results manually:
  git add eval/results/
  git commit -m "chore(eval): prod smoke run ${TS} @ ${HW_TAG}"
  git push origin $(git rev-parse --abbrev-ref HEAD)

Or rerun as \`bash eval/run_prod_smoke.sh --push\` to commit + push automatically.
EOF
fi

# overall exit
[[ ${PYTEST_RC} -eq 0 && ${SCEN1_RC} -eq 0 && ${SCEN2_RC} -eq 0 && ${SCEN3_RC} -eq 0 && ${ACC_RC} -eq 0 ]] && exit 0 || exit 1
