#!/usr/bin/env bash
# =============================================================================
# Phase 3 — stuck CPU engine introspection (hang-proof, 외부 timeout wrapped)
#
# 최상위에서 `exec timeout --foreground` 로 자기 자신을 감싸 process group
# 전체 kill 을 보장. 내부 명령이 kernel stuck (D state) 이어도 무조건 종료.
#
# 도구: ps / /proc stack,wchan / py-spy --nonblocking / perf stat
# perf record + report 는 사용 안 함 (DWARF/symbol 해석 hang 주범).
# =============================================================================
set -uo pipefail

# -----------------------------------------------------------------------------
# 최상위 timeout wrapper — process group 전체 kill 보장
# 첫 실행 시 timeout 밑으로 다시 exec. DEADLINE 후 SIGTERM + 5초 → SIGKILL.
# _PHASE3_WRAPPED=1 이면 이미 wrap 된 상태라 skip.
# -----------------------------------------------------------------------------
DEADLINE_OUTER="${DEADLINE:-60}"
if [[ "${_PHASE3_WRAPPED:-0}" != "1" ]]; then
    export _PHASE3_WRAPPED=1
    exec timeout --foreground --kill-after=5 --signal=TERM "${DEADLINE_OUTER}" \
        bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${OUT_DIR:-}" ]]; then
    TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
    OUT_DIR="${SCRIPT_DIR}/results/${TS}/phase3"
fi
mkdir -p "${OUT_DIR}"

PERF_STAT_SECS="${PERF_STAT_SECS:-5}"
PYSPY_TIMEOUT="${PYSPY_TIMEOUT:-8}"

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

# safe_timeout: SIGTERM → 2초 후 SIGKILL
safe_timeout() { timeout --kill-after=2 --signal=TERM "$@"; }

log "=== Phase 3 시작 (외부 timeout ${DEADLINE_OUTER}s wrapped) ==="
log "결과        : ${OUT_DIR}"

# 의존성
HAVE_PYSPY=0; HAVE_PERF=0
command -v py-spy >/dev/null 2>&1 && HAVE_PYSPY=1
command -v perf   >/dev/null 2>&1 && HAVE_PERF=1
(( HAVE_PYSPY )) || log "[WARN] py-spy 없음"
(( HAVE_PERF  )) || log "[WARN] perf 없음"

# 대상 PID
mapfile -t PIDS < <(pgrep -f CPU_EngineCore 2>/dev/null)
if [[ ${#PIDS[@]} -eq 0 ]]; then
    log "[ERROR] CPU_EngineCore 프로세스 없음"
    exit 1
fi
log "CPU_EngineCore : ${#PIDS[@]} 개 ${PIDS[*]}"

# ----------------------------------------------------------------------------
# summary 헤더
# ----------------------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.md"
{
    echo "# Phase 3 — $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')"
    echo
    echo "## 대상"
    for PID in "${PIDS[@]}"; do
        cmd=$(ps -p ${PID} -o comm= 2>/dev/null)
        nthr=$(grep -E '^Threads' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        aff=$(grep -E '^Cpus_allowed_list' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        echo "- PID ${PID}  ${cmd}  threads=${nthr}  cpus=${aff}"
    done
} > "${SUMMARY}"

# ----------------------------------------------------------------------------
# 캡처 함수들 — 오직 hang-proof 도구만 사용
# ----------------------------------------------------------------------------

# (a) ps + /proc/<tid>/{stack,wchan} — 즉시 반환 보장 (kernel read)
capture_threads() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_threads.txt"
    {
        echo "### ps -L + kernel wchan/stack"
        ps -L -p ${PID} -o tid,stat,psr,pcpu,comm --no-headers 2>/dev/null \
            | sort -k4 -nr > "/tmp/_threads_${PID}_$$.tmp"
        total=$(wc -l < "/tmp/_threads_${PID}_$$.tmp")
        echo "전체 thread : ${total}"
        echo
        echo "상태별:"
        awk '{print $2}' "/tmp/_threads_${PID}_$$.tmp" | sort | uniq -c | sort -rn
        echo
        echo "%CPU > 30 threads:"
        awk '$4+0 > 30' "/tmp/_threads_${PID}_$$.tmp"
        echo
        echo "Top-20 (+ wchan + kernel stack head-3):"
        head -20 "/tmp/_threads_${PID}_$$.tmp" | while read tid stat psr pcpu comm; do
            wchan=$(cat "/proc/${tid}/wchan" 2>/dev/null | tr -d '\0' || echo '?')
            echo "  tid=${tid} ${stat} cpu${psr} ${pcpu}% ${comm}  wchan=${wchan:-0}"
            if [[ -r "/proc/${tid}/stack" ]]; then
                safe_timeout 2 awk '{printf "    %s\n", $2}' "/proc/${tid}/stack" 2>/dev/null | head -3
            fi
        done
        rm -f "/tmp/_threads_${PID}_$$.tmp"
    } > "${OUT}" 2>&1
}

# (b) py-spy dump --nonblocking — process 안 멈춤 + 외부 timeout
capture_pyspy() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_pyspy.txt"
    {
        echo "### py-spy dump --nonblocking (timeout ${PYSPY_TIMEOUT}s)"
        if (( HAVE_PYSPY )); then
            safe_timeout ${PYSPY_TIMEOUT} py-spy dump --pid ${PID} --nonblocking 2>&1 \
                || echo "[FAIL or timeout]"
        else
            echo "[SKIP] py-spy 미설치"
        fi
    } > "${OUT}" 2>&1
}

# (c) perf stat — 정확히 N초 후 exit. perf record/report 절대 사용 안 함.
#     계수값만 수집 → compute-bound vs memory-bound 판별
capture_perf_stat() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_perf.txt"
    {
        echo "### perf stat (counters, ${PERF_STAT_SECS}s)"
        echo "note: 심볼 해석 없이 counter 만 — compute/memory-bound 판별용"
        echo
        if (( HAVE_PERF )); then
            safe_timeout $((PERF_STAT_SECS + 3)) \
                perf stat -p ${PID} \
                -e cycles,instructions,cache-references,cache-misses,branch-misses,context-switches,cpu-migrations,task-clock \
                -- sleep ${PERF_STAT_SECS} 2>&1 \
                || echo "[FAIL or timeout]"
        else
            echo "[SKIP] perf 미설치"
        fi
    } > "${OUT}" 2>&1
}

# ----------------------------------------------------------------------------
# 실행 — 모든 capture 를 최상위 shell 의 직계 자식으로 (pkill -P $$ 가 도달)
# ----------------------------------------------------------------------------
log "=== 병렬 캡처 시작 (perf_stat=${PERF_STAT_SECS}s, pyspy=${PYSPY_TIMEOUT}s) ==="
T0=$(date +%s)

CHILD_PIDS=()
for PID in "${PIDS[@]}"; do
    capture_threads   "${PID}" & CHILD_PIDS+=($!)
    capture_pyspy     "${PID}" & CHILD_PIDS+=($!)
    capture_perf_stat "${PID}" & CHILD_PIDS+=($!)
done

# 각 child 에 개별 wait (watchdog 이 DEADLINE 후 모두 kill 하므로 hang 불가)
for p in "${CHILD_PIDS[@]}"; do
    wait "${p}" 2>/dev/null || true
done

T1=$(date +%s)
log "=== 캡처 완료 ($((T1 - T0))s) ==="

# ----------------------------------------------------------------------------
# summary 마무리
# ----------------------------------------------------------------------------
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
    echo "- py-spy 가 Python attention / block_table 함수 → **B (GIL)**"
    echo "- py-spy \`<native>\` + threads 상위 tid 의 wchan 이 \`futex_wait\` → native lock 경쟁"
    echo "- perf stat 의 \`instructions/cycles\` (IPC) < 0.5 → memory-bound 확정"
    echo "- perf stat 의 \`cache-misses/cache-references\` > 20% → cache unfriendly (긴 KV 지목)"
    echo
    echo "**소요: $((T1 - T0))s**"
} >> "${SUMMARY}"

log "=== Phase 3 완료 ==="
log "summary : ${SUMMARY}"
