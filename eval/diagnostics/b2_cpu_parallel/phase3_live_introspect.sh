#!/usr/bin/env bash
# =============================================================================
# Phase 3 — 최소 구성: py-spy dump 만. 외부 timeout 30초 wrap.
#
# 목적은 CPU engine 의 Python stack 을 잡는 것 하나뿐. perf 제거, watchdog
# 제거, 복잡한 trap 제거. 외부 timeout 이 process group 전체 kill 보장.
#
# 사용 (단독):
#   bash eval/diagnostics/b2_cpu_parallel/phase3_live_introspect.sh
#
# OUT_DIR env 로 출력 위치 override (run_all.sh 에서 지정).
# =============================================================================

# 최상위 timeout wrap — 재진입 시 skip
# --foreground 는 쓰지 않음. --foreground 가 있으면 timeout 이 process group
# 전체가 아니라 직계 자식에만 signal 을 보내서 손자 (py-spy 등) 가 살아남음.
# 실측 확인됨 (테스트 3 vs 4): --foreground 없이 process group kill 이 유일한 보장.
if [[ "${_PHASE3_WRAPPED:-0}" != "1" ]]; then
    export _PHASE3_WRAPPED=1
    exec timeout --kill-after=5 --signal=TERM 30 bash "$0" "$@"
fi

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${OUT_DIR:-}" ]]; then
    TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
    OUT_DIR="${SCRIPT_DIR}/results/${TS}/phase3"
fi
mkdir -p "${OUT_DIR}"

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

log "=== Phase 3 시작 (외부 timeout 30s) ==="
log "결과: ${OUT_DIR}"

# py-spy 확인
if ! command -v py-spy >/dev/null 2>&1; then
    log "[WARN] py-spy 미설치. pip install py-spy 후 재시도"
    echo "py-spy not installed" > "${OUT_DIR}/ERROR.txt"
    exit 0
fi

# CPU engine PID
mapfile -t PIDS < <(pgrep -f CPU_EngineCore 2>/dev/null)
if [[ ${#PIDS[@]} -eq 0 ]]; then
    log "[ERROR] CPU_EngineCore 프로세스 없음"
    echo "no CPU_EngineCore process" > "${OUT_DIR}/ERROR.txt"
    exit 0
fi
log "target: ${#PIDS[@]} engines (${PIDS[*]})"

# 기본 정보 + OMP 라이브러리 체크 + py-spy 덤프 + kernel stack
for PID in "${PIDS[@]}"; do
    OUT="${OUT_DIR}/engine_${PID}_pyspy.txt"
    {
        echo "### PID ${PID} — $(ps -p ${PID} -o comm= 2>/dev/null)"
        echo "threads: $(grep -E '^Threads' /proc/${PID}/status 2>/dev/null | awk '{print $2}')"
        echo "cpus   : $(grep -E '^Cpus_allowed_list' /proc/${PID}/status 2>/dev/null | awk '{print $2}')"
        echo
        echo "### OMP/BLAS 라이브러리 로드 상태 (duplication 의심 — 318 thread 원인?)"
        echo "# libgomp + libiomp5 양쪽 다 있으면 OMP runtime 중복 → pool 2~3배"
        grep -E 'libomp|libgomp|libiomp|libmkl_|libopenblas|libblis' /proc/${PID}/maps 2>/dev/null \
            | awk '{print $NF}' | sort -u | tee /tmp/phase3_omp_${PID}_$$.txt \
            || echo "(maps 접근 실패)"
        echo
        echo "### Thread 이름 분포 (어느 pool 에서 왔는지)"
        # ps -L 의 comm 은 15 char 잘림. /proc/<tid>/comm 이 정확.
        for tid in $(ls /proc/${PID}/task/ 2>/dev/null); do
            cat /proc/${PID}/task/${tid}/comm 2>/dev/null
        done | sort | uniq -c | sort -rn | head -20
        echo
        echo "### OMP 환경변수 (subprocess 에 꽂힌 값)"
        tr '\0' '\n' < /proc/${PID}/environ 2>/dev/null \
            | grep -E '^(OMP_|MKL_|KMP_|IPEX_|OPENBLAS_|VLLM_CPU_|VLLM_HYBRID_)' \
            | sort
        echo
        echo "### ps -L top-10 by %CPU (with state)"
        ps -L -p ${PID} -o tid,stat,psr,pcpu,comm --no-headers 2>/dev/null \
            | sort -k4 -nr | tee /tmp/phase3_threads_${PID}_$$.txt | head -10
        echo
        echo "### py-spy dump (without --nonblocking; SIGSTOP ~100ms for consistent read)"
        echo "# 주의: Python 3.12 + --nonblocking 조합에서 'Failed to copy PyCodeObject'"
        echo "# 발생 → --nonblocking 제거. ptrace 로 잠시 정지 후 정확한 Python stack 덤프."
        if ! timeout --kill-after=2 12 py-spy dump --pid ${PID} 2>&1 ; then
            echo
            echo "### fallback — py-spy dump --nonblocking"
            timeout --kill-after=2 10 py-spy dump --pid ${PID} --nonblocking 2>&1
        fi
        echo
        echo "### kernel stack — top-5 by %CPU (from ps above)"
        echo "# main thread 는 아마도 Rl+ state. 다른 thread 는 Sl+ (sleep) 에서"
        echo "# 어떤 futex/lock/syscall 에서 대기 중인지 보여줌."
        head -5 /tmp/phase3_threads_${PID}_$$.txt | while read tid stat psr pcpu comm; do
            echo
            echo "---- tid=${tid} stat=${stat} cpu${psr} pcpu=${pcpu} ----"
            wchan=$(cat "/proc/${tid}/wchan" 2>/dev/null | tr -d '\0' || echo '?')
            echo "wchan: ${wchan:-0}"
            if [[ -r "/proc/${tid}/stack" ]]; then
                echo "stack:"
                timeout --kill-after=1 3 awk '{print "  " $2}' "/proc/${tid}/stack" 2>/dev/null | head -15
            else
                echo "stack: [/proc/${tid}/stack 읽기 권한 없음 (root 필요)]"
            fi
        done
        rm -f /tmp/phase3_threads_${PID}_$$.txt /tmp/phase3_omp_${PID}_$$.txt
    } > "${OUT}" 2>&1 &
done
wait

log "=== Phase 3 완료 ==="
log "파일: ${OUT_DIR}/engine_*_pyspy.txt"
ls "${OUT_DIR}/"
