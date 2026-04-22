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

# 기본 정보 + py-spy 덤프 (engine 별 파일)
for PID in "${PIDS[@]}"; do
    OUT="${OUT_DIR}/engine_${PID}_pyspy.txt"
    {
        echo "### PID ${PID} — $(ps -p ${PID} -o comm= 2>/dev/null)"
        echo "threads: $(grep -E '^Threads' /proc/${PID}/status 2>/dev/null | awk '{print $2}')"
        echo "cpus   : $(grep -E '^Cpus_allowed_list' /proc/${PID}/status 2>/dev/null | awk '{print $2}')"
        echo
        echo "### ps -L top-10 by %CPU"
        ps -L -p ${PID} -o tid,stat,psr,pcpu,comm --no-headers 2>/dev/null \
            | sort -k4 -nr | head -10
        echo
        echo "### py-spy dump --nonblocking"
        timeout --kill-after=2 10 py-spy dump --pid ${PID} --nonblocking 2>&1
    } > "${OUT}" &
done
wait

log "=== Phase 3 완료 ==="
log "파일: ${OUT_DIR}/engine_*_pyspy.txt"
ls "${OUT_DIR}/"
