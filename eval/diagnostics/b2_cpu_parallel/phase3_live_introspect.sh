#!/usr/bin/env bash
# =============================================================================
# Phase 3 — stuck CPU engine 실시간 introspection
#
# heavy hybrid 가 돌고 있는 동안 CPU_EngineCore 프로세스들의
#   1. Python call stack (py-spy dump)
#   2. native hot function (perf record 10s)
#   3. thread state 분포 (ps -L)
# 를 캡처해서 snapshots/<ts>/ 아래 저장.
#
# 사용 (heavy 서버 돌고 있는 상태에서):
#   bash eval/diagnostics/b2_cpu_parallel/phase3_live_introspect.sh
#
# 의존성 (자동 확인 + 없으면 메시지):
#   - py-spy (pip install py-spy)  — Python stack
#   - perf (linux-tools)            — native hot function
#
# 출력:
#   eval/diagnostics/b2_cpu_parallel/snapshots/<YYYYMMDD_HHMMSS>/
#     engine_<pid>_pyspy.txt
#     engine_<pid>_perf.txt
#     engine_<pid>_threads.txt
#     summary.md
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
OUT_DIR="${SCRIPT_DIR}/snapshots/${TS}"
mkdir -p "${OUT_DIR}"

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

log "=== Phase 3 introspection 시작 ==="
log "출력        : ${OUT_DIR}"

# ----------------------------------------------------------------------------
# 0. 의존성 확인
# ----------------------------------------------------------------------------
HAVE_PYSPY=0
HAVE_PERF=0
command -v py-spy >/dev/null 2>&1 && HAVE_PYSPY=1
command -v perf   >/dev/null 2>&1 && HAVE_PERF=1

if (( ! HAVE_PYSPY )); then
    log "[WARN] py-spy 없음. 설치: pip install py-spy"
fi
if (( ! HAVE_PERF )); then
    log "[WARN] perf 없음. 설치: apt install linux-tools-common linux-tools-generic"
fi

# ----------------------------------------------------------------------------
# 1. CPU_EngineCore pid 수집
# ----------------------------------------------------------------------------
mapfile -t PIDS < <(pgrep -f CPU_EngineCore 2>/dev/null)
if [[ ${#PIDS[@]} -eq 0 ]]; then
    log "[ERROR] CPU_EngineCore 프로세스 없음. heavy 서버 살아 있는지 확인"
    exit 1
fi
log "CPU_EngineCore 감지: ${#PIDS[@]} 개 (${PIDS[*]})"

# ----------------------------------------------------------------------------
# 2. summary.md 초기화
# ----------------------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.md"
{
    echo "# Phase 3 live introspection — ${TS} (KST)"
    echo
    echo "## 대상 프로세스"
    echo
    for PID in "${PIDS[@]}"; do
        CMD=$(ps -p ${PID} -o comm= 2>/dev/null)
        AFF=$(grep -E '^Cpus_allowed_list' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        NTHR=$(grep -E '^Threads' /proc/${PID}/status 2>/dev/null | awk '{print $2}')
        echo "- PID ${PID}  (${CMD})  — threads=${NTHR}  cpus=${AFF}"
    done
    echo
    echo "## 수집 항목"
} > "${SUMMARY}"

# ----------------------------------------------------------------------------
# 3. Per-engine capture
# ----------------------------------------------------------------------------
for PID in "${PIDS[@]}"; do
    log "--- PID ${PID} 캡처 ---"

    # (a) thread state breakdown (상시 가능)
    THREADS_FILE="${OUT_DIR}/engine_${PID}_threads.txt"
    {
        echo "### thread state breakdown (ps -L)"
        echo "legend: R=Running  S=Sleeping  D=Uninterruptible  T=Stopped"
        echo
        ps -L -p ${PID} -o tid,stat,psr,pcpu,comm --no-headers 2>/dev/null \
            | sort -k4 -nr > /tmp/threads_${PID}.tmp
        echo "전체 thread 수 : $(wc -l < /tmp/threads_${PID}.tmp)"
        echo
        echo "상태별 집계:"
        awk '{print $2}' /tmp/threads_${PID}.tmp | sort | uniq -c | sort -rn
        echo
        echo "%CPU > 30 threads (active):"
        awk '$4+0 > 30' /tmp/threads_${PID}.tmp
        echo
        echo "(top-20 by %CPU):"
        head -20 /tmp/threads_${PID}.tmp
        rm /tmp/threads_${PID}.tmp
    } > "${THREADS_FILE}"
    log "  [ok] threads → $(basename ${THREADS_FILE})"

    # (b) py-spy dump
    PYSPY_FILE="${OUT_DIR}/engine_${PID}_pyspy.txt"
    if (( HAVE_PYSPY )); then
        {
            echo "### py-spy dump (Python call stack of all threads)"
            echo
            py-spy dump --pid ${PID} 2>&1 || echo "[FAIL] py-spy returned error"
        } > "${PYSPY_FILE}"
        log "  [ok] py-spy → $(basename ${PYSPY_FILE})"
    else
        echo "[SKIP] py-spy 미설치" > "${PYSPY_FILE}"
    fi

    # (c) perf record 10s
    PERF_FILE="${OUT_DIR}/engine_${PID}_perf.txt"
    if (( HAVE_PERF )); then
        PERF_DATA="/tmp/perf_${PID}.data"
        {
            echo "### perf top (10 sec sampling)"
            echo "note: kernel symbols require kptr_restrict=0"
            echo
            perf record -p ${PID} -g -F 99 -o ${PERF_DATA} -- sleep 10 2>&1 | tail -5
            echo
            echo "--- perf report (head 50 lines) ---"
            perf report -i ${PERF_DATA} --stdio --sort dso,symbol 2>&1 | head -60
            rm -f ${PERF_DATA}
        } > "${PERF_FILE}"
        log "  [ok] perf → $(basename ${PERF_FILE})"
    else
        echo "[SKIP] perf 미설치" > "${PERF_FILE}"
    fi
done

# ----------------------------------------------------------------------------
# 4. summary.md 채움
# ----------------------------------------------------------------------------
{
    echo
    echo "### 판정 가이드"
    echo
    echo "- py-spy stack 에 **attention forward / block table walk / Python loop** 잡힘"
    echo "  → **B 가설** (Python-level serialization / GIL)"
    echo
    echo "- py-spy stack 에 \`<native>\` 만 (Python idle) 잡히고,"
    echo "  perf 에서 \`ipex_*_paged_attention\` / \`paged_attention_v1\` 등이 hot"
    echo "  → **A 가설** (native kernel 이 long-ctx 에서 single-thread)"
    echo
    echo "- py-spy stack 이 torch \`sdpa\` 쪽에 있거나 perf 가 \`at::native::sdpa\`"
    echo "  → **C 가설** (sdpa_loop fallback)"
    echo
    echo "- threads.txt 에서 **대부분 S(sleeping) + 소수 R(running)** 이면"
    echo "  스레드 풀은 살아있지만 스케줄 되지 않는 것. 상위 R thread 의 stack 을 py-spy 로 확인."
} >> "${SUMMARY}"

log "=== Phase 3 완료 ==="
log "요약        : ${SUMMARY}"
log "전체 파일   : ls ${OUT_DIR}"

ls -la "${OUT_DIR}"
