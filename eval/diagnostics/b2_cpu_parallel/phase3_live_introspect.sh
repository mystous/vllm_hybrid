#!/usr/bin/env bash
# =============================================================================
# Phase 3 — py-spy record (flame graph) + 프로세스 info
#
# 이전 버전 실패 원인:
#   engine 1개에 py-spy dump + record 3종을 동시에 ptrace attach 시도 →
#   ptrace 는 프로세스당 tracer 1개만 허용 → 모두 실패.
#
# 이 버전:
#   Step 1  비-ptrace 정보 (ps / /proc / OMP) — 모든 engine 병렬, 즉시
#   Step 2  py-spy record --native (engine 별 1개) — 다른 PID 들은 병렬 가능
#   Step 3  py-spy dump --nonblocking — record 끝난 후 (ptrace 해제됨)
#
# 외부 timeout 90s, record 30s + 여유 60s.
# =============================================================================

if [[ "${_PHASE3_WRAPPED:-0}" != "1" ]]; then
    export _PHASE3_WRAPPED=1
    exec timeout --kill-after=5 --signal=TERM 90 bash "$0" "$@"
fi

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${OUT_DIR:-}" ]]; then
    TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
    OUT_DIR="${SCRIPT_DIR}/results/${TS}/phase3"
fi
mkdir -p "${OUT_DIR}"

RECORD_DURATION="${RECORD_DURATION:-30}"

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

log "=== Phase 3 시작 (외부 timeout 90s) ==="
log "결과: ${OUT_DIR}"

if ! command -v py-spy >/dev/null 2>&1; then
    log "[WARN] py-spy 미설치"
    echo "py-spy not installed" > "${OUT_DIR}/ERROR.txt"
    exit 0
fi

mapfile -t PIDS < <(pgrep -f CPU_EngineCore 2>/dev/null)
if [[ ${#PIDS[@]} -eq 0 ]]; then
    log "[ERROR] CPU_EngineCore 프로세스 없음"
    echo "no CPU_EngineCore process" > "${OUT_DIR}/ERROR.txt"
    exit 0
fi
log "target: ${#PIDS[@]} engines (${PIDS[*]})"

# =============================================================================
# Step 1 — 비-ptrace 정보 수집 (engine 별 병렬, 즉시)
# ps -L / /proc/maps (OMP libs) / /proc/task/comm / /proc/environ /
# /proc/tid/{wchan,stack} — 모두 read-only 이며 ptrace 관여 없음.
# =============================================================================
log "Step 1: 비-ptrace 정보 수집 (parallel)"

collect_info() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_info.txt"
    local TMP_THREADS="/tmp/phase3_threads_${PID}_$$.tmp"
    {
        echo "### PID ${PID} — $(ps -p ${PID} -o comm= 2>/dev/null)"
        echo "threads: $(grep -E '^Threads' /proc/${PID}/status 2>/dev/null | awk '{print $2}')"
        echo "cpus   : $(grep -E '^Cpus_allowed_list' /proc/${PID}/status 2>/dev/null | awk '{print $2}')"
        echo
        echo "### OMP/BLAS 라이브러리 로드 상태 (duplication 체크)"
        grep -E 'libomp|libgomp|libiomp|libmkl_|libopenblas|libblis' /proc/${PID}/maps 2>/dev/null \
            | awk '{print $NF}' | sort -u \
            || echo "(maps 접근 실패)"
        echo
        echo "### Thread 이름 분포"
        for tid in $(ls /proc/${PID}/task/ 2>/dev/null); do
            cat /proc/${PID}/task/${tid}/comm 2>/dev/null
        done | sort | uniq -c | sort -rn | head -20
        echo
        echo "### Subprocess env (/proc/environ 은 spawn 시점 값)"
        tr '\0' '\n' < /proc/${PID}/environ 2>/dev/null \
            | grep -E '^(OMP_|MKL_|KMP_|IPEX_|OPENBLAS_|VLLM_CPU_|VLLM_HYBRID_)' \
            | sort \
            || echo "(environ 접근 실패 또는 없음)"
        echo
        echo "### ps -L top-10 by %CPU"
        ps -L -p ${PID} -o tid,stat,psr,pcpu,comm --no-headers 2>/dev/null \
            | sort -k4 -nr | tee "${TMP_THREADS}" | head -10
        echo
        echo "### kernel wchan/stack — top-5 threads"
        head -5 "${TMP_THREADS}" | while read tid stat psr pcpu comm; do
            echo
            echo "---- tid=${tid} stat=${stat} cpu${psr} pcpu=${pcpu} ----"
            wchan=$(cat "/proc/${tid}/wchan" 2>/dev/null | tr -d '\0' || echo '?')
            echo "wchan: ${wchan:-0}"
            if [[ -r "/proc/${tid}/stack" ]]; then
                echo "stack:"
                timeout --kill-after=1 2 awk '{print "  " $2}' "/proc/${tid}/stack" 2>/dev/null | head -10
            fi
        done
        rm -f "${TMP_THREADS}"
    } > "${OUT}" 2>&1
}

for PID in "${PIDS[@]}"; do
    collect_info "${PID}" &
done
wait
log "  Step 1 완료"

# =============================================================================
# Step 2 — py-spy record --native (engine 별 1 회, 서로 다른 PID 는 병렬 가능)
# 같은 PID 에 여러 ptrace 금지 → engine 당 하나만. --native 로 C 스택 포함.
# --native 실패 시 (libunwind 등 이슈) --native 없이 fallback.
# =============================================================================
log "Step 2: py-spy record --native ${RECORD_DURATION}s (engine 간 parallel)"

record_one() {
    local PID=$1
    local SVG="${OUT_DIR}/engine_${PID}_flame.svg"
    local LOG="${OUT_DIR}/engine_${PID}_flame.log"
    {
        echo "=== py-spy record --native ${RECORD_DURATION}s ==="
        if timeout --kill-after=5 $((RECORD_DURATION + 15)) \
                py-spy record -p ${PID} -d ${RECORD_DURATION} --native \
                -f flamegraph -o "${SVG}" 2>&1; then
            echo "[ok] flame graph saved"
        else
            echo "[FAIL] --native. retry without --native..."
            rm -f "${SVG}"
            if timeout --kill-after=5 $((RECORD_DURATION + 15)) \
                    py-spy record -p ${PID} -d ${RECORD_DURATION} \
                    -f flamegraph -o "${SVG}" 2>&1; then
                echo "[ok] flame graph (no native) saved"
            else
                echo "[FAIL] record without --native too"
            fi
        fi
    } > "${LOG}" 2>&1
}

for PID in "${PIDS[@]}"; do
    record_one "${PID}" &
done
wait
log "  Step 2 완료"

# =============================================================================
# Step 3 — py-spy dump (record 끝난 후 — 이제 ptrace 해제됨)
# 단일 snapshot. record 결과 (flame graph) 를 보완.
# =============================================================================
log "Step 3: py-spy dump (record 완료 후)"

dump_one() {
    local PID=$1 OUT="${OUT_DIR}/engine_${PID}_dump.txt"
    {
        echo "### py-spy dump (record 완료 후 — ptrace 해제 상태)"
        if ! timeout --kill-after=2 10 py-spy dump --pid ${PID} 2>&1 ; then
            echo
            echo "### fallback: dump --nonblocking"
            timeout --kill-after=2 8 py-spy dump --pid ${PID} --nonblocking 2>&1
        fi
    } > "${OUT}" 2>&1
}

for PID in "${PIDS[@]}"; do
    dump_one "${PID}" &
done
wait
log "  Step 3 완료"

# =============================================================================
# 결과 정리
# =============================================================================
log "=== Phase 3 완료 ==="
log "파일:"
ls -la "${OUT_DIR}/"
