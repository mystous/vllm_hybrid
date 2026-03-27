#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — GPU-only / Hybrid 풀 평가 파이프라인
#
# 실행 순서:
#   1. GPU-only 서버 시작
#   2. 모니터 시작 (GPU/CPU utilization)
#   3. 서버 준비 대기
#   4. 벤치마크 실행
#   5. 모니터 종료, 서버 종료
#   6. Hybrid 서버 시작
#   7. 모니터 시작
#   8. 서버 준비 대기
#   9. 벤치마크 실행
#  10. 모니터 종료, 서버 종료
#  11. 비교 리포트 생성
#
# 사용법:
#   ./run_eval.sh              # GPU-only + Hybrid 모두 실행
#   ./run_eval.sh gpu          # GPU-only만 실행
#   ./run_eval.sh hybrid       # Hybrid만 실행
#   ./run_eval.sh compare      # 기존 결과로 비교만 실행
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] .env 파일을 찾을 수 없습니다: $ENV_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

RESULTS_DIR="${SCRIPT_DIR}/${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

MODE="${1:-all}"  # all / gpu / hybrid / compare

SERVER_PID=""
MONITOR_PID=""

# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local url="http://localhost:${PORT}/health"
    local timeout="${SERVER_READY_TIMEOUT:-300}"
    local poll="${SERVER_READY_POLL:-3}"
    local elapsed=0

    log "서버 준비 대기 중... (최대 ${timeout}초)"
    while ! curl -sf "$url" > /dev/null 2>&1; do
        if [[ $elapsed -ge $timeout ]]; then
            log "[ERROR] 서버 시작 타임아웃 (${timeout}초 초과)"
            return 1
        fi
        sleep "$poll"
        elapsed=$((elapsed + poll))
        log "  대기 중... ${elapsed}/${timeout}초"
    done
    log "서버 준비 완료 (${elapsed}초 소요)"
}

start_server() {
    local server_mode="$1"
    log "=== 서버 시작: MODE=${server_mode} ==="
    bash "${SCRIPT_DIR}/serve.sh" "$server_mode" \
        > "${RESULTS_DIR}/${server_mode}_serve.log" 2>&1 &
    SERVER_PID=$!
    log "서버 PID: ${SERVER_PID}"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "서버 종료 (PID=${SERVER_PID})"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    # 혹시 남은 vLLM 프로세스 정리
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
}

start_monitor() {
    local prefix="$1"
    local interval="${MONITOR_INTERVAL:-1}"
    log "모니터 시작: prefix=${prefix}"
    python3 "${SCRIPT_DIR}/monitor.py" "$prefix" --interval "$interval" \
        > "${RESULTS_DIR}/monitor_${prefix##*/}.log" 2>&1 &
    MONITOR_PID=$!
    log "모니터 PID: ${MONITOR_PID}"
}

stop_monitor() {
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        log "모니터 종료 (PID=${MONITOR_PID})"
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

cleanup() {
    log "정리 중..."
    stop_monitor
    stop_server
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# GPU-only 평가
# ---------------------------------------------------------------------------

run_gpu_only() {
    log "=========================================="
    log " [1/2] GPU-only 평가 시작"
    log "=========================================="

    stop_server   # 혹시 남은 서버 정리

    start_server "gpu"
    start_monitor "${RESULTS_DIR}/gpu_only_monitor"

    wait_for_server

    log "--- GPU-only 벤치마크 실행 ---"
    bash "${SCRIPT_DIR}/benchmark.sh" "gpu_only"

    stop_monitor
    stop_server

    log "GPU-only 평가 완료."
}

# ---------------------------------------------------------------------------
# Hybrid 평가
# ---------------------------------------------------------------------------

run_hybrid() {
    log "=========================================="
    log " [2/2] Hybrid 평가 시작"
    log "=========================================="

    stop_server

    start_server "hybrid"
    start_monitor "${RESULTS_DIR}/hybrid_monitor"

    wait_for_server

    log "--- Hybrid 벤치마크 실행 ---"
    bash "${SCRIPT_DIR}/benchmark.sh" "hybrid"

    stop_monitor
    stop_server

    log "Hybrid 평가 완료."
}

# ---------------------------------------------------------------------------
# 비교 리포트
# ---------------------------------------------------------------------------

run_compare() {
    log "=========================================="
    log " 비교 리포트 생성"
    log "=========================================="
    python3 "${SCRIPT_DIR}/compare.py" \
        --results-dir "${RESULTS_DIR}" \
        --gpu-label gpu_only \
        --hybrid-label hybrid
}

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

log "eval 시작: MODE=${MODE}"
log "결과 경로: ${RESULTS_DIR}"
log "모델: ${MODEL}"

case "$MODE" in
    all)
        run_gpu_only
        run_hybrid
        run_compare
        ;;
    gpu)
        run_gpu_only
        ;;
    hybrid)
        run_hybrid
        ;;
    compare)
        run_compare
        ;;
    *)
        echo "Usage: $0 [all|gpu|hybrid|compare]" >&2
        exit 1
        ;;
esac

log "완료."
