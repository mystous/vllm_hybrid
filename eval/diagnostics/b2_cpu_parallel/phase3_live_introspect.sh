#!/usr/bin/env bash
# =============================================================================
# Phase 3 — stuck CPU engine 실시간 introspection (defensive version)
#
# 이전 버전이 `perf record -g` + `perf report --sort dso,symbol` 조합으로
# DWARF call graph 해석에 수 분 걸리는 문제 → 가볍고 빠른 도구로 교체.
#
# 모든 외부 명령에 hard timeout + SIGKILL fallback.
# engine 간 완전 병렬. 전체 deadline 90초. py-spy 는 --nonblocking.
#
# 사용:
#   bash eval/diagnostics/b2_cpu_parallel/phase3_live_introspect.sh
#
# env override:
#   OUT_DIR=<dir>         결과 저장 위치 (run_all.sh 에서 지정)
#   PERF_DURATION=10      perf top 샘플 지속 (초)
#   PYSPY_TIMEOUT=10      py-spy 최대 대기
#   DEADLINE=90           전체 스크립트 hard deadline
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${OUT_DIR:-}" ]]; then
    TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
    OUT_DIR="${SCRIPT_DIR}/results/${TS}/phase3"
fi
mkdir -p "${OUT_DIR}"

PERF_DURATION="${PERF_DURATION:-10}"
PYSPY_TIMEOUT="${PYSPY_TIMEOUT:-10}"
DEADLINE="${DEADLINE:-90}"

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

# 안전한 timeout 래퍼 — SIGTERM → 2초 후 SIGKILL
safe_timeout() {
    local sec=$1; shift
    timeout --kill-after=2 --signal=TERM "${sec}" "$@"
}

log "=== Phase 3 시작 (deadline ${DEADLINE}s) ==="
log "결과        : ${OUT_DIR}"

# 의존성
HAVE_PYSPY=0; HAVE_PERF=0
command -v py-spy >/dev/null 2>&1 && HAVE_PYSPY=1
command -v perf   >/dev/null 2>&1 && HAVE_PERF=1
(( HAVE_PYSPY )) || log "[WARN] py-spy 없음 (pip install py-spy) — Python stack skip"
(( HAVE_PERF  )) || log "[WARN] perf 없음 (apt install linux-tools) — native hot skip"

# PID 수집
mapfile -t PIDS < <(pgrep -f CPU_EngineCore 2>/dev/null)
if [[ ${#PIDS[@]} -eq 0 ]]; then
    log "[ERROR] CPU_EngineCore 프로세스 없음"
    exit 1
fi
log "CPU_EngineCore : ${#PIDS[@]} 개 ${PIDS[*]}"

# -----------------------------------------------------------------------------
# summary.md 초기화
# -----------------------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.md"
{
    echo "# Phase 3 live introspection — $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')"
    echo
    echo "## 대상 프로세스"
    echo
    for PID in "${PIDS[@]}"; do
        CMD=$(ps -p ${PID} -o comm= 2>/dev/null)
        AFF=$(grep -E '^Cpus_allowed_list' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        NTHR=$(grep -E '^Threads' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        echo "- PID ${PID}  (${CMD})  threads=${NTHR}  cpus=${AFF}"
    done
} > "${SUMMARY}"

# -----------------------------------------------------------------------------
# 캡처 함수들 — 각각 엄격한 timeout
# -----------------------------------------------------------------------------

capture_threads() {
    # ps -L 은 항상 빠름 (ms). 추가로 /proc/<tid>/stack 로 커널 stack 까지.
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_threads.txt"
    {
        echo "### thread state + kernel stack (top-20 by %CPU)"
        echo "legend: R=Running  S=Sleeping  D=Uninterruptible  T=Stopped"
        echo
        ps -L -p ${PID} -o tid,stat,psr,pcpu,comm --no-headers 2>/dev/null \
            | sort -k4 -nr > /tmp/threads_${PID}.tmp
        echo "전체 thread 수: $(wc -l < /tmp/threads_${PID}.tmp)"
        echo
        echo "상태별 집계:"
        awk '{print $2}' /tmp/threads_${PID}.tmp | sort | uniq -c | sort -rn
        echo
        echo "%CPU > 30:"
        awk '$4+0 > 30' /tmp/threads_${PID}.tmp
        echo
        echo "Top-20 by %CPU (+ kernel stack):"
        head -20 /tmp/threads_${PID}.tmp | while read tid stat psr pcpu comm; do
            echo "  tid=${tid} stat=${stat} cpu${psr} ${pcpu}% ${comm}"
            # /proc/<tid>/stack 는 root 가 대개 필요. 실패해도 무시
            if [[ -r "/proc/${tid}/stack" ]]; then
                awk '{printf "    %s\n", $2}' "/proc/${tid}/stack" 2>/dev/null | head -5
            fi
        done
        rm -f /tmp/threads_${PID}.tmp
    } > "${OUT}"
}

capture_pyspy() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_pyspy.txt"
    {
        echo "### py-spy dump (nonblocking, timeout=${PYSPY_TIMEOUT}s)"
        echo
        if (( HAVE_PYSPY )); then
            # --nonblocking: process 를 멈추지 않음 → 빠름
            safe_timeout ${PYSPY_TIMEOUT} py-spy dump --pid ${PID} --nonblocking 2>&1 \
                || echo "[FAIL or timeout] py-spy"
        else
            echo "[SKIP] py-spy 미설치"
        fi
    } > "${OUT}"
}

capture_perf_top() {
    # perf top 을 perf record 대신 사용:
    # - call graph 없음 (-g 없이) → DWARF 해석 없음
    # - 샘플링 중 즉시 누적 → perf.data 파일 파싱 불필요
    # - --stdio --no-children 로 단순 심볼 랭킹
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_perf.txt"
    {
        echo "### perf top (${PERF_DURATION}s, no call-graph)"
        echo
        if (( HAVE_PERF )); then
            # 방법 A — perf top live (하드 timeout 로 끊음)
            safe_timeout $((PERF_DURATION + 5)) perf top \
                -p ${PID} --stdio --no-children -F 99 2>&1 \
                | head -50 \
                || echo "[FAIL or timeout] perf top"
            echo
            echo "### perf stat ($(expr ${PERF_DURATION} / 2)s counters)"
            echo
            safe_timeout $((PERF_DURATION / 2 + 5)) perf stat \
                -p ${PID} -- sleep $((PERF_DURATION / 2)) 2>&1 \
                || echo "[FAIL or timeout] perf stat"
        else
            echo "[SKIP] perf 미설치"
        fi
    } > "${OUT}"
}

# -----------------------------------------------------------------------------
# 병렬 캡처 — engine 간 + engine 내부 모두 parallel
# -----------------------------------------------------------------------------
log "=== 병렬 캡처 (perf=${PERF_DURATION}s, pyspy=${PYSPY_TIMEOUT}s) ==="
T0=$(date +%s)

(
    for PID in "${PIDS[@]}"; do
        (
            capture_threads "${PID}" &
            capture_pyspy   "${PID}" &
            capture_perf_top "${PID}" &
            wait
            log "  PID ${PID} 캡처 완료"
        ) &
    done
    wait
) &
CAPTURE_PID=$!

# 전체 deadline 적용 — 그 안에 못 끝나면 kill
if ! safe_timeout "${DEADLINE}" bash -c "wait ${CAPTURE_PID}" 2>/dev/null; then
    log "[WARN] deadline ${DEADLINE}s 초과 — 남은 프로세스 강제 종료"
    kill -9 ${CAPTURE_PID} 2>/dev/null || true
    # perf / py-spy 자식 프로세스도 정리
    pkill -9 -P ${CAPTURE_PID} 2>/dev/null || true
fi

T1=$(date +%s)
log "=== 캡처 종료 (elapsed $((T1 - T0))s) ==="

# -----------------------------------------------------------------------------
# summary.md 마무리
# -----------------------------------------------------------------------------
{
    echo
    echo "## 캡처 결과 파일"
    echo
    for PID in "${PIDS[@]}"; do
        echo "### PID ${PID}"
        for kind in threads pyspy perf; do
            f="engine_${PID}_${kind}.txt"
            if [[ -f "${OUT_DIR}/${f}" ]]; then
                sz=$(stat -c '%s' "${OUT_DIR}/${f}" 2>/dev/null || echo '?')
                echo "- [${f}](${f}) (${sz} bytes)"
            fi
        done
        echo
    done
    echo "## 판정 가이드"
    echo
    echo "- py-spy stack 이 Python attention 함수 (forward / block table) → **B (GIL)**"
    echo "- py-spy \`<native>\` + perf top 상위가 \`ipex_*paged_attention\` → **A (IPEX)**"
    echo "- py-spy \`torch::sdpa\` → **C (sdpa_loop)**"
    echo "- threads.txt 대부분 S + 소수 R → 스케줄되지 못함, 상위 R 의 kernel stack 확인"
    echo
    echo "**전체 소요: $((T1 - T0))s**"
} >> "${SUMMARY}"

log "=== Phase 3 완료 ==="
log "summary     : ${SUMMARY}"
ls -la "${OUT_DIR}/"
