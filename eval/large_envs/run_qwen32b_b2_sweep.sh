#!/usr/bin/env bash
# =============================================================================
# run_qwen32b_b2_sweep.sh — B2 검증용 Qwen2.5-32B 전량 sweep
#
# 각 env 에 대해 gpu_only + hybrid sweep (seqs 1/2/4/8/16) 순차 실행.
# 결과는 eval/results/<ts>_... 에 저장됨 (수동 mv 는 사용자 몫).
#
# 사용법 (vllm_hybrid 루트에서):
#   bash eval/large_envs/run_qwen32b_b2_sweep.sh
#
# env 를 override 하려면:
#   ENVS="eval/envs/foo.env eval/large_envs/bar.env" \
#     bash eval/large_envs/run_qwen32b_b2_sweep.sh
#
# seqs 목록 변경:
#   SEQS_LIST="1 4 16" bash eval/large_envs/run_qwen32b_b2_sweep.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# 기본값 (환경변수로 override 가능)
# -----------------------------------------------------------------------------
ENVS="${ENVS:-eval/large_envs/g0_h100x8_qwen32b_longctx_16k_16k.env}"
# 00_tp8 는 완료됨. 전량 재실행 필요 시:
#   ENVS="eval/envs/g0_h100x8_qwen32b_00_tp8.env eval/large_envs/g0_h100x8_qwen32b_longctx_16k_16k.env" \
#     bash eval/large_envs/run_qwen32b_b2_sweep.sh
SEQS_LIST="${SEQS_LIST:-1 2 4 8 16}"
RUN_ENV="${RUN_ENV:-/tmp/run.env}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"  # 32B + long ctx 로딩 오래 걸림

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

log "=== B2 sweep 시작 ==="
log "ENVS     : ${ENVS}"
log "SEQS_LIST: ${SEQS_LIST}"
log "RUN_ENV  : ${RUN_ENV}"

# 기존 서버 사전 정리
pkill -f api_server 2>/dev/null || true
pkill -f 'serve\.sh' 2>/dev/null || true
sleep 3

wait_ready() {
    local elapsed=0
    while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        if (( elapsed > READY_TIMEOUT )); then
            log "[ERROR] ready timeout ${READY_TIMEOUT}s 초과"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "server ready (${elapsed}s)"
}

cleanup_server() {
    local spid="$1"
    [[ -n "${spid}" ]] && kill "${spid}" 2>/dev/null
    wait "${spid}" 2>/dev/null || true
    pkill -f api_server 2>/dev/null || true
    pkill -f 'serve\.sh' 2>/dev/null || true
    sleep 10
}

for ENV_SRC in ${ENVS}; do
    if [[ ! -f "${ENV_SRC}" ]]; then
        log "[WARN] env 파일 없음: ${ENV_SRC}, skip"
        continue
    fi
    cp "${ENV_SRC}" "${RUN_ENV}"
    TAG=$(basename "${ENV_SRC}" .env)

    # -------------------------------------------------------------------------
    # 1) gpu_only
    # -------------------------------------------------------------------------
    log "=== [${TAG}] gpu_only start ==="
    ./eval/serve.sh gpu_only "${RUN_ENV}" > "/tmp/srv_${TAG}_gpu.log" 2>&1 &
    SPID=$!
    if ! wait_ready; then
        cleanup_server "${SPID}"
        log "[${TAG}] gpu_only READY 실패, skip to next env"
        continue
    fi
    ./eval/bench.sh gpu_only "${RUN_ENV}" || log "[WARN] bench rc=$?"
    cleanup_server "${SPID}"
    log "=== [${TAG}] gpu_only done ==="

    # -------------------------------------------------------------------------
    # 2) hybrid seqs sweep
    # -------------------------------------------------------------------------
    for SEQS in ${SEQS_LIST}; do
        sed -i "s/^HYBRID_CPU_MAX_SEQS=.*/HYBRID_CPU_MAX_SEQS=${SEQS}/" "${RUN_ENV}"
        log "=== [${TAG}] hybrid seqs=${SEQS} start ==="
        ./eval/serve.sh hybrid "${RUN_ENV}" > "/tmp/srv_${TAG}_h${SEQS}.log" 2>&1 &
        SPID=$!
        if ! wait_ready; then
            cleanup_server "${SPID}"
            log "[${TAG}] hybrid seqs=${SEQS} READY 실패, skip to next seqs"
            continue
        fi
        ./eval/bench.sh hybrid "${RUN_ENV}" || log "[WARN] bench rc=$?"
        cleanup_server "${SPID}"
        log "=== [${TAG}] hybrid seqs=${SEQS} done ==="
    done
done

log "=== ALL_DONE ==="
log "최근 결과: eval/results/ 에서 상위 $(echo ${ENVS} | wc -w) × $(( $(echo ${SEQS_LIST} | wc -w) + 1 )) 개 확인"
log "ls -1t eval/results/ | head"
