#!/usr/bin/env bash
# =============================================================================
# Phase 3 — py-spy dump (snapshot) + record (30s flame graph)
#
# 목적: "어느 함수 / 어느 C 호출 chain 에서 시간 쓰는지" 를 30초 연속 샘플링
# 으로 확정. 단일 dump 로는 1 순간 스냅샷 뿐이라 우연성 있음 → record 병행.
#
# 수집물 (engine 별):
#   engine_<pid>_pyspy.txt         dump snapshot + ps + kernel stack + OMP check
#   engine_<pid>_flame.svg         Python flame graph (30s)
#   engine_<pid>_flame_native.svg  Native + Python flame graph (30s, libtorch/IPEX/libgomp)
#   engine_<pid>_raw.txt           Raw samples (greppable)
#
# 외부 timeout 90s wrap. record 30s + 여유 60s.
# =============================================================================

# 최상위 timeout wrap — 재진입 시 skip
# --foreground 는 쓰지 않음 — process group kill 보장 위해.
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

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

RECORD_DURATION="${RECORD_DURATION:-30}"

log "=== Phase 3 시작 (외부 timeout 90s, record ${RECORD_DURATION}s) ==="
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

# -----------------------------------------------------------------------------
# py-spy record — 30초 연속 샘플링 → flame graph (SVG) + raw (text)
# dump (스냅샷) 과 병행. 모든 engine × 3 종 record 를 parallel 로 실행.
# -----------------------------------------------------------------------------
for PID in "${PIDS[@]}"; do
    # Python-only flame graph
    timeout --kill-after=5 $((RECORD_DURATION + 15)) \
        py-spy record -p ${PID} -d ${RECORD_DURATION} \
        -f flamegraph -o "${OUT_DIR}/engine_${PID}_flame.svg" \
        > "${OUT_DIR}/engine_${PID}_flame.log" 2>&1 &

    # Native + Python flame graph (C extensions: libtorch / IPEX / libgomp)
    # --native 는 libunwind 를 사용. 디버깅 심볼 없으면 주소로 나올 수 있지만
    # dso 이름 (libtorch.so 등) 은 식별됨.
    timeout --kill-after=5 $((RECORD_DURATION + 15)) \
        py-spy record -p ${PID} -d ${RECORD_DURATION} --native \
        -f flamegraph -o "${OUT_DIR}/engine_${PID}_flame_native.svg" \
        > "${OUT_DIR}/engine_${PID}_flame_native.log" 2>&1 &

    # Raw samples (grep/sort 가능한 텍스트)
    timeout --kill-after=5 $((RECORD_DURATION + 15)) \
        py-spy record -p ${PID} -d ${RECORD_DURATION} \
        -f raw -o "${OUT_DIR}/engine_${PID}_raw.txt" \
        > "${OUT_DIR}/engine_${PID}_raw.log" 2>&1 &
done

wait

log "=== Phase 3 완료 ==="
log "파일:"
ls -la "${OUT_DIR}/"
