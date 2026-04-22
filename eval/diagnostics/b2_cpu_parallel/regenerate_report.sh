#!/usr/bin/env bash
# =============================================================================
# regenerate_report.sh — 기존 phase3 결과 디렉토리의 FINAL_REPORT.md 재생성
#
# 초기 run_all.sh 의 report 생성 로직이 옛 파일명 (phase3/summary.md) 을
# 찾아서 "Phase 3 결과 없음" 을 출력하던 버그 후처리용. 새 phase3 형식
# (engine_*_flame.svg + engine_*_info.txt + engine_*_dump.txt) 을 읽어
# FINAL_REPORT.md 를 재생성.
#
# 사용:
#   bash eval/diagnostics/b2_cpu_parallel/regenerate_report.sh <results_dir>
#   # 예: bash eval/diagnostics/b2_cpu_parallel/regenerate_report.sh \
#   #         eval/diagnostics/b2_cpu_parallel/results/20260422_063129
#
# 인자 생략 시 최신 디렉토리 자동 선택.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ge 1 ]]; then
    RESULTS_DIR="$1"
else
    RESULTS_DIR=$(ls -dt "${SCRIPT_DIR}/results"/*/ 2>/dev/null | head -1 | sed 's:/$::')
fi

if [[ -z "${RESULTS_DIR}" || ! -d "${RESULTS_DIR}" ]]; then
    echo "[ERROR] 결과 디렉토리 없음: ${RESULTS_DIR:-<empty>}"
    exit 1
fi

PHASE1_DIR="${RESULTS_DIR}/phase1"
PHASE2_DIR="${RESULTS_DIR}/phase2"
PHASE3_DIR="${RESULTS_DIR}/phase3"
REPORT="${RESULTS_DIR}/FINAL_REPORT.md"
TS=$(basename "${RESULTS_DIR}")

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "Regenerating: ${REPORT}"
echo "  Phase 1: ${PHASE1_DIR}"
echo "  Phase 2: ${PHASE2_DIR}"
echo "  Phase 3: ${PHASE3_DIR}"

{
    echo "# B2 CPU Parallelism 검증 통합 보고서"
    echo
    echo "- 실행 시각 (KST): ${TS}"
    echo "- 결과 디렉토리: \`${RESULTS_DIR#${REPO_ROOT}/}\`"
    echo "- (재생성 시각: $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'))"
    echo

    # ------------------------------------------------------------------------
    # Phase 1 — dispatch
    # ------------------------------------------------------------------------
    echo "## 1. Phase 1 — 정적 dispatch 분석"
    echo
    if [[ -f "${PHASE1_DIR}/dispatch_static.txt" ]]; then
        echo "파일: [\`phase1/dispatch_static.txt\`](phase1/dispatch_static.txt)"
        echo
        echo "\`\`\`"
        grep -E 'Section A|Section B|\[.*\]  called|L [0-9]+ \|' \
            "${PHASE1_DIR}/dispatch_static.txt" 2>/dev/null | head -40
        echo "\`\`\`"
    else
        echo "없음"
    fi
    echo

    # ------------------------------------------------------------------------
    # Phase 3 — Live introspection (새 파일 형식 지원)
    # ------------------------------------------------------------------------
    echo "## 2. Phase 3 — Live introspection"
    echo
    if compgen -G "${PHASE3_DIR}/engine_*_flame.svg" > /dev/null \
      || compgen -G "${PHASE3_DIR}/engine_*_info.txt"  > /dev/null; then
        for SVG in "${PHASE3_DIR}"/engine_*_flame.svg; do
            [[ -f "${SVG}" ]] || continue
            EPID=$(basename "${SVG}" | sed -E 's/engine_([0-9]+)_flame\.svg/\1/')
            INFO="${PHASE3_DIR}/engine_${EPID}_info.txt"
            DUMP="${PHASE3_DIR}/engine_${EPID}_dump.txt"
            echo
            echo "### Engine PID ${EPID}"
            echo
            echo "- Flame graph: [\`phase3/engine_${EPID}_flame.svg\`](phase3/engine_${EPID}_flame.svg) (브라우저에서 열기)"
            [[ -f "${INFO}" ]] && echo "- Info: [\`phase3/engine_${EPID}_info.txt\`](phase3/engine_${EPID}_info.txt)"
            [[ -f "${DUMP}" ]] && echo "- Dump: [\`phase3/engine_${EPID}_dump.txt\`](phase3/engine_${EPID}_dump.txt)"
            echo

            if [[ -f "${INFO}" ]]; then
                echo "#### Process 요약"
                echo "\`\`\`"
                sed -n '1,3p' "${INFO}"
                echo
                awk '/### OMP\/BLAS/,/^$/' "${INFO}" | head -10
                awk '/### Thread 이름 분포/,/^$/' "${INFO}" | head -10
                awk '/### ps -L top-10/,/^$/' "${INFO}" | head -15
                echo "\`\`\`"
                echo
            fi

            echo "#### Top hot functions (60 초 flame graph 샘플 상위 20)"
            echo "\`\`\`"
            grep -oE '<title>[^<]+\([0-9]+ samples,[^<]+</title>' "${SVG}" \
                | sed -E 's/<\/?title>//g' \
                | python3 -c "
import sys, re
rows = []
for line in sys.stdin:
    m = re.search(r'\((\d+) samples', line)
    if m:
        rows.append((int(m.group(1)), line.strip()))
rows.sort(key=lambda r: -r[0])
for n, line in rows[:20]:
    print(f'{n:5d}  {line}')
" 2>/dev/null || echo "(flame graph 파싱 실패)"
            echo "\`\`\`"
            echo
        done
    elif [[ -f "${PHASE3_DIR}/summary.md" ]]; then
        # legacy format
        cat "${PHASE3_DIR}/summary.md"
    else
        echo "Phase 3 결과 없음"
    fi
    echo

    # ------------------------------------------------------------------------
    # Phase 2 — TRACE counter
    # ------------------------------------------------------------------------
    echo "## 3. Phase 2 — TRACE counter 실측"
    echo
    if [[ -f "${PHASE2_DIR}/trace_counters.txt" ]]; then
        echo "파일: [\`phase2/trace_counters.txt\`](phase2/trace_counters.txt)"
        echo
        echo "\`\`\`"
        cat "${PHASE2_DIR}/trace_counters.txt"
        echo "\`\`\`"
    else
        echo "Phase 2 결과 없음"
    fi
    echo

    # ------------------------------------------------------------------------
    # 판정 참고
    # ------------------------------------------------------------------------
    echo "## 4. 판정 참고"
    echo
    cat <<'EOF'
Top hot functions 를 본 뒤, 아래 범주 중 지배적인 것 식별:

| 주요 hot path | 의미 |
|---|---|
| `find_longest_cache_hit` / `get_computed_blocks` 가 dominant | **prefix cache 탐색** 이 bottleneck. heavy workload 에서 16K prompt × 1024 hash block 의 Python loop 매칭 |
| `_update_states` 가 dominant | GPU 상속 state update 의 PP 분기 + block_ids loop 가 heavy 에서 비용 |
| `execute_model` (gpu_model_runner) 가 dominant 이지만 내부가 C++ (forward) | 정상 compute, 최적화 대상 아님 |
| `decorate_context` / `__enter__` 같은 context manager 가 많음 | inclusive time (stack 공통), self-time 아님 |
| IPEX kernel 이 dominant | native kernel 최적화 대상 |

해석은 `super_power/draft/B2/` 문서에 반영.
EOF
} > "${REPORT}"

echo
echo "[OK] Report regenerated:"
echo "  ${REPORT}"
wc -l "${REPORT}"
