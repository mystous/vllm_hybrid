#!/usr/bin/env bash
# =============================================================================
# Phase 3 — stuck CPU engine 실시간 introspection (robust)
#
# 전 버전의 문제:
#   - `perf top --stdio` 가 TTY 입력 대기로 hang
#   - deadline `wait` 이 새 shell 에서 안 먹혀서 강제종료 안 됨
#
# 이 버전:
#   - perf record (-g 없이) + perf report → 확실히 exit
#   - py-spy --nonblocking + 외부 timeout
#   - 전체 script 시작 시 watchdog subshell 이 DEADLINE 후 pkill -P $$
#   - trap EXIT 으로 어떤 종료 경로에서도 자식 정리
#
# 사용: bash eval/diagnostics/b2_cpu_parallel/phase3_live_introspect.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${OUT_DIR:-}" ]]; then
    TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
    OUT_DIR="${SCRIPT_DIR}/results/${TS}/phase3"
fi
mkdir -p "${OUT_DIR}"

PERF_DURATION="${PERF_DURATION:-5}"
PYSPY_TIMEOUT="${PYSPY_TIMEOUT:-10}"
DEADLINE="${DEADLINE:-60}"

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

# -----------------------------------------------------------------------------
# Watchdog — DEADLINE 후 모든 자식 강제 종료
# -----------------------------------------------------------------------------
(
    sleep "${DEADLINE}"
    echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] [DEADLINE] ${DEADLINE}s 초과 — 강제 종료"
    pkill -9 -P $$ 2>/dev/null
) &
WATCHDOG_PID=$!

# 종료 시 watchdog + 자식들 정리
cleanup() {
    kill "${WATCHDOG_PID}" 2>/dev/null
    pkill -P $$ 2>/dev/null
    rm -f /tmp/perf_$$.data /tmp/perf_*_$$.data /tmp/threads_*_$$.tmp
}
trap cleanup EXIT INT TERM

log "=== Phase 3 시작 (deadline ${DEADLINE}s) ==="
log "결과        : ${OUT_DIR}"

# 의존성
HAVE_PYSPY=0; HAVE_PERF=0
command -v py-spy >/dev/null 2>&1 && HAVE_PYSPY=1
command -v perf   >/dev/null 2>&1 && HAVE_PERF=1
(( HAVE_PYSPY )) || log "[WARN] py-spy 없음"
(( HAVE_PERF  )) || log "[WARN] perf 없음"

# PID
mapfile -t PIDS < <(pgrep -f CPU_EngineCore 2>/dev/null)
if [[ ${#PIDS[@]} -eq 0 ]]; then
    log "[ERROR] CPU_EngineCore 프로세스 없음"
    exit 1
fi
log "CPU_EngineCore : ${#PIDS[@]} 개 ${PIDS[*]}"

# -----------------------------------------------------------------------------
# summary 시작
# -----------------------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.md"
{
    echo "# Phase 3 — $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')"
    echo
    echo "## 대상"
    for PID in "${PIDS[@]}"; do
        local_cmd=$(ps -p ${PID} -o comm= 2>/dev/null)
        local_nthr=$(grep -E '^Threads' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        local_aff=$(grep -E '^Cpus_allowed_list' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        echo "- PID ${PID}  ${local_cmd}  threads=${local_nthr}  cpus=${local_aff}"
    done
} > "${SUMMARY}"

# -----------------------------------------------------------------------------
# 각 엔진 capture — 엄격한 외부 timeout
# -----------------------------------------------------------------------------
# safe_timeout: SIGTERM 이후 2초 → SIGKILL
safe_timeout() { timeout --kill-after=2 --signal=TERM "$@"; }

capture_threads() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_threads.txt"
    {
        echo "### thread state + kernel wchan/stack (top-20 by %CPU)"
        ps -L -p ${PID} -o tid,stat,psr,pcpu,comm --no-headers 2>/dev/null \
            | sort -k4 -nr > /tmp/threads_${PID}_$$.tmp
        echo "전체 thread : $(wc -l < /tmp/threads_${PID}_$$.tmp)"
        echo
        echo "상태별:"
        awk '{print $2}' /tmp/threads_${PID}_$$.tmp | sort | uniq -c | sort -rn
        echo
        echo "%CPU > 30:"
        awk '$4+0 > 30' /tmp/threads_${PID}_$$.tmp
        echo
        echo "Top-20 (+ kernel stack head-3):"
        head -20 /tmp/threads_${PID}_$$.tmp | while read tid stat psr pcpu comm; do
            wchan=$(cat "/proc/${tid}/wchan" 2>/dev/null || echo '?')
            echo "  tid=${tid} ${stat} cpu${psr} ${pcpu}% ${comm}  wchan=${wchan}"
            if [[ -r "/proc/${tid}/stack" ]]; then
                awk '{printf "    %s\n", $2}' "/proc/${tid}/stack" 2>/dev/null | head -3
            fi
        done
        rm -f /tmp/threads_${PID}_$$.tmp
    } > "${OUT}" 2>&1
}

capture_pyspy() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_pyspy.txt"
    {
        echo "### py-spy dump --nonblocking (timeout ${PYSPY_TIMEOUT}s)"
        if (( HAVE_PYSPY )); then
            safe_timeout ${PYSPY_TIMEOUT} py-spy dump --pid ${PID} --nonblocking 2>&1 \
                || echo "[FAIL or timed out]"
        else
            echo "[SKIP] py-spy 미설치"
        fi
    } > "${OUT}" 2>&1
}

capture_perf() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_perf.txt"
    local DATA="/tmp/perf_${PID}_$$.data"
    {
        echo "### perf record (no call-graph) + report — ${PERF_DURATION}s sampling"
        if (( HAVE_PERF )); then
            # -g 없이 PC 샘플만 → report 시 DWARF 해석 없음
            safe_timeout $((PERF_DURATION + 5)) \
                perf record -p ${PID} -F 99 -o "${DATA}" -- sleep ${PERF_DURATION} 2>&1 \
                | tail -5
            if [[ -f "${DATA}" ]]; then
                echo
                echo "--- top 30 symbols ---"
                safe_timeout 10 \
                    perf report -i "${DATA}" --stdio --no-children --sort symbol 2>&1 \
                    | head -40 \
                    || echo "[FAIL or timed out] perf report"
                rm -f "${DATA}"
            else
                echo "[FAIL] perf record 결과 파일 없음"
            fi
            echo
            echo "### perf stat (counters, $((PERF_DURATION / 2))s)"
            safe_timeout $((PERF_DURATION / 2 + 5)) \
                perf stat -p ${PID} -- sleep $((PERF_DURATION / 2)) 2>&1 \
                || echo "[FAIL or timed out] perf stat"
        else
            echo "[SKIP] perf 미설치"
        fi
    } > "${OUT}" 2>&1
}

# -----------------------------------------------------------------------------
# 실행 — engine 간 + engine 내부 병렬
# -----------------------------------------------------------------------------
log "=== 병렬 캡처 (perf=${PERF_DURATION}s, pyspy=${PYSPY_TIMEOUT}s) ==="
T0=$(date +%s)

for PID in "${PIDS[@]}"; do
    (
        capture_threads "${PID}" &
        capture_pyspy   "${PID}" &
        capture_perf    "${PID}" &
        wait
        log "  PID ${PID} 완료"
    ) &
done
wait

T1=$(date +%s)
log "=== 캡처 완료 ($((T1 - T0))s) ==="

# -----------------------------------------------------------------------------
# summary 마무리
# -----------------------------------------------------------------------------
{
    echo
    echo "## 파일"
    for PID in "${PIDS[@]}"; do
        echo "### PID ${PID}"
        for kind in threads pyspy perf; do
            f="engine_${PID}_${kind}.txt"
            [[ -f "${OUT_DIR}/${f}" ]] && echo "- [${f}](${f})"
        done
    done
    echo
    echo "## 판정"
    echo "- py-spy 가 Python attention/block_table 에서 잡힘 → **B (GIL)**"
    echo "- py-spy \`<native>\` + perf 상위 \`ipex_*paged_attention\` → **A (IPEX)**"
    echo "- py-spy \`torch::sdpa\` 또는 perf \`at::native::sdpa\` → **C (sdpa_loop)**"
    echo "- threads 대부분 S + 소수 R, R 스레드의 wchan 이 futex/mutex → native lock"
    echo
    echo "**소요: $((T1 - T0))s**"
} >> "${SUMMARY}"

log "=== Phase 3 완료 ==="
log "summary : ${SUMMARY}"

# watchdog 정리
kill "${WATCHDOG_PID}" 2>/dev/null
