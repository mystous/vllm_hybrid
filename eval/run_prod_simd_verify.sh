#!/usr/bin/env bash
# =============================================================================
# run_prod_simd_verify.sh — TSK_003 §4.2a/§4.2b prod 검증 + 결과 자동 저장.
#
# 풀 prod_smoke (~15-20 분) 의 [1/5] pytest + [5/5] e2e accuracy 만 떼어낸
# 빠른 검증판 (~8-10 분). 목적은 두 가지:
#
#   1) TST_004 numerical cross-check — portable C++ vs AVX-512 / portable C++
#      vs AMX BF16 의 결과가 BF16 round-off tolerance 안인지 확인. dev 에서
#      cpuid 게이트 (AVX-512 fuse-off / AMX hardware 미지원) 로 자동 skipif
#      된 80 케이스가 prod 의 Sapphire Rapids+ 에서 처음 활성됨. SIMD kernel
#      의 numerical correctness 의 게이트.
#
#   2) Phase 4c e2e — SIMD-built 환경에서 cold path dispatcher 가 정상 발화
#      하고 D-ii 가 binding gate 로 PASS 하는지. split-on-only 모드라 baseline
#      LLM 로딩을 생략해 baseline 비교 대신 verdict 안정성 + suspicious_no
#      _cold_path detector 미발동만 본다 (정식 비교는 풀 prod_smoke 에서).
#
# Usage:
#   bash eval/run_prod_simd_verify.sh             # run + save (push manually)
#   bash eval/run_prod_simd_verify.sh --push      # run + commit + push
#
# Output layout:
#   eval/results/<TS>_<HW_TAG>_simd_verify/
#     ├── README.md              # meta + verdict 요약 + 분석 지점
#     ├── isa_info.txt           # /proc/cpuinfo + nvidia-smi + ulimit + numa
#     ├── tst004_pytest.log      # TST_004 stdout + skip 사유
#     ├── tst004_junit.xml       # JUnit XML — 분석 시 PASS/FAIL 자동 파싱
#     └── e2e_quick.log          # split-on-only 실행 로그 (대용량 dispatch
#                                # 진단 라인 포함)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_simd_verify"
mkdir -p "${OUT_DIR}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python)"
fi

PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --------------------------------------------------------------------- meta

log "writing meta to ${OUT_DIR}/README.md"
{
    echo "# ${TS}_${HW_TAG}_simd_verify"
    echo
    echo "- timestamp: ${TS}"
    echo "- hw_tag:    ${HW_TAG}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo
    echo "## components"
    echo "- pytest TST_004 cross-check (TSK_003 §4.2a portable vs AVX-512 + §4.2b portable vs AMX)"
    echo "    └─ 40 + 40 = 80 케이스. dev 에서는 cpuid 게이트로 자동 skipif"
    echo "- eval/run_e2e_accuracy.py --split-on-only (cold-path dispatcher 발화 + D-ii binding 확인)"
    echo "    └─ baseline LLM 로딩 생략. 8 prompts × 14336 input × 32 output 의 짧은 verifier"
    echo
    echo "본 결과 디렉토리는 prod 검증 후 dev 에서 분석하기 위해 push 되며,"
    echo "tst004_junit.xml 의 PASS/FAIL 패턴 + e2e_quick.log 의 [IDE_006 diag ...]"
    echo "발화 카운터를 통해 SIMD kernel 의 numerical correctness 와 dispatcher"
    echo "wiring 안정성을 분석한다. 정식 throughput / accuracy 비교는 풀"
    echo "run_prod_smoke.sh (--push) 가 별도로 수행."
} > "${OUT_DIR}/README.md"

log "capturing CPU/GPU/NUMA snapshot to ${OUT_DIR}/isa_info.txt"
{
    echo "=== /proc/cpuinfo (ISA flags) ==="
    grep -oE 'avx512[a-z_0-9]*|amx_[a-z_]*|avx2|sse4_2' /proc/cpuinfo | sort -u
    echo
    echo "=== uname -r (kernel; AMX requires ≥ 5.16) ==="
    uname -r
    echo
    echo "=== ulimit -l (locked memory; needed for pinned alloc) ==="
    ulimit -l
    echo
    echo "=== free -h (system RAM) ==="
    free -h
    echo
    echo "=== nvidia-smi ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>&1 || echo "nvidia-smi unavailable"
    echo
    echo "=== compiler ==="
    c++ --version 2>&1 | head -1
    echo
    echo "=== numactl --hardware ==="
    numactl --hardware 2>&1 || echo "numactl unavailable"
} > "${OUT_DIR}/isa_info.txt"

# --------------------------------------------------------------------- 1) TST_004

log "[1/2] pytest TST_004 cross-check (portable vs AVX-512 / portable vs AMX)"
TST_RC=0
"${PYTHON}" -m pytest \
    tests/v1/cpu_partial_attention/test_avx512_cross_check.py \
    tests/v1/cpu_partial_attention/test_amx_cross_check.py \
    -v --tb=short \
    --junit-xml="${OUT_DIR}/tst004_junit.xml" \
    2>&1 | tee "${OUT_DIR}/tst004_pytest.log" || TST_RC=$?
log "  TST_004 exit=${TST_RC}"

# --------------------------------------------------------------------- 2) split-on-only e2e quick

log "[2/2] e2e accuracy (split-on-only, cold path dispatcher firing verify)"
E2E_RC=0
HW_TAG="${HW_TAG}" "${PYTHON}" "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/vllm_original_long_ctx.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_long_ctx.env" \
    --split-on-only \
    --max-prompts 8 \
    --max-tokens 32 \
    --logprobs 0 \
    --output-dir "${OUT_DIR}/e2e_quick_artifacts" \
    2>&1 | tee "${OUT_DIR}/e2e_quick.log" || E2E_RC=$?
log "  e2e quick exit=${E2E_RC}"

# --------------------------------------------------------------------- summary

# Pull verdict signals out of the logs so the README is self-contained.
TST004_PASSED=$(grep -oE '[0-9]+ passed' "${OUT_DIR}/tst004_pytest.log" | head -1 || echo "")
TST004_FAILED=$(grep -oE '[0-9]+ failed' "${OUT_DIR}/tst004_pytest.log" | head -1 || echo "")
TST004_SKIPPED=$(grep -oE '[0-9]+ skipped' "${OUT_DIR}/tst004_pytest.log" | head -1 || echo "")
E2E_DI=$(grep -oE 'D-i  \(token divergence\):  (PASS|FAIL)' "${OUT_DIR}/e2e_quick.log" | head -1 || echo "(N/A)")
E2E_DII=$(grep -oE 'D-ii \(logprob / PPL\):     (PASS|FAIL)' "${OUT_DIR}/e2e_quick.log" | head -1 || echo "(N/A)")
E2E_OVERALL=$(grep -oE 'overall:                  (PASS|FAIL)' "${OUT_DIR}/e2e_quick.log" | head -1 || echo "(N/A)")

{
    echo
    echo "## exit codes"
    echo "- TST_004 pytest:                ${TST_RC}"
    echo "- e2e quick (split-on-only):     ${E2E_RC}"
    echo
    echo "## verdict signals"
    echo "- TST_004 cross-check counts:    ${TST004_PASSED} ${TST004_FAILED} ${TST004_SKIPPED}"
    echo "- e2e quick D-i:                 ${E2E_DI}"
    echo "- e2e quick D-ii:                ${E2E_DII}"
    echo "- e2e quick overall:             ${E2E_OVERALL}"
    echo
    echo "## interpretation cheat-sheet"
    echo "- 80 passed, 0 failed: SIMD kernel numerical correctness 확정 → throughput sweep 진입 가능"
    echo "- N failed (특히 AMX): tst004_junit.xml 의 첫 FAIL stack trace + 파라미터 분석으로 즉시 fix"
    echo "- 80 skipped: cpuid 게이트가 AVX-512/AMX 활성을 못 잡은 것 — isa_info.txt 의 /proc/cpuinfo 확인"
    echo "- e2e overall PASS + suspicious_no_cold_path=False: dispatcher wiring 정상"
    echo "- e2e overall FAIL + suspicious_no_cold_path=True: cold path silent bypass — 추가 진단 필요"
} >> "${OUT_DIR}/README.md"

log "artifacts -> ${OUT_DIR}"
log "key logs:"
log "  - ${OUT_DIR}/README.md           (meta + verdict 요약)"
log "  - ${OUT_DIR}/tst004_pytest.log   (pytest stdout)"
log "  - ${OUT_DIR}/tst004_junit.xml    (JUnit XML — 자동 파싱용)"
log "  - ${OUT_DIR}/e2e_quick.log       (e2e dispatch 진단 포함)"
log "  - ${OUT_DIR}/e2e_quick_artifacts (split_on.json + comparison.json + README.md)"
log "  - ${OUT_DIR}/isa_info.txt        (CPU / NUMA / kernel 스냅샷)"

# --------------------------------------------------------------------- push

if [[ ${PUSH} -eq 1 ]]; then
    log "staging eval/results/ + git push"
    git add eval/results/
    if git diff --cached --quiet; then
        log "  nothing to commit"
    else
        git commit -m "chore(eval): TSK_003 SIMD prod 검증 결과 ${TS} @ ${HW_TAG}"
        BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        git push origin "${BRANCH}"
    fi
else
    cat <<EOF

[push hint] To push the results manually:
  git add eval/results/
  git commit -m "chore(eval): TSK_003 SIMD prod 검증 결과 ${TS} @ ${HW_TAG}"
  git push origin $(git rev-parse --abbrev-ref HEAD)

Or rerun as \`bash eval/run_prod_simd_verify.sh --push\` to commit + push automatically.
EOF
fi

# overall exit
[[ ${TST_RC} -eq 0 && ${E2E_RC} -eq 0 ]] && exit 0 || exit 1
