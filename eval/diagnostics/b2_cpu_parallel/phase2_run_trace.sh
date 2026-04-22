#!/usr/bin/env bash
# =============================================================================
# Phase 2 — TRACE=1 켠 짧은 heavy run (10~20분)
#
# decode path counter ([HYBRID-CPU-ATTN]) 를 로그에 찍어서
# heavy input shape 이 어느 path (custom_avx/ipex/sdpa_batched/sdpa_loop) 를
# 타는지 실측 확정.
#
# 사용:
#   bash eval/diagnostics/b2_cpu_parallel/phase2_run_trace.sh
#
# 출력:
#   - eval/results/<ts>_H_*_seqs2/hybrid_server_run.log 에 [HYBRID-CPU-ATTN] 라인
#   - 스크립트 말미에서 해당 라인을 요약 추출해서 stdout 에 프린트
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

ENV_SRC="${SCRIPT_DIR}/g0_h100x8_qwen32b_longctx_trace.env"
RUN_ENV="/tmp/run_phase2.env"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

if [[ ! -f "${ENV_SRC}" ]]; then
    log "[ERROR] env 없음: ${ENV_SRC}"
    exit 1
fi

# 기존 서버 정리
pkill -f api_server 2>/dev/null || true
pkill -f 'serve\.sh' 2>/dev/null || true
sleep 3

cp "${ENV_SRC}" "${RUN_ENV}"
TAG="phase2_trace"

wait_ready() {
    local elapsed=0
    while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        if (( elapsed > READY_TIMEOUT )); then
            log "[ERROR] ready timeout ${READY_TIMEOUT}s"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        (( elapsed % 30 == 0 )) && log "  waiting ready... (${elapsed}s)"
    done
    log "server ready (${elapsed}s)"
}

log "=== Phase 2 시작 ==="
log "env         : ${ENV_SRC}"
log "workload    : 16K input / 256 output / 4 prompts / conc 4"
log "tracing     : VLLM_HYBRID_TRACE=1 (매 decode call log)"

./eval/serve.sh hybrid "${RUN_ENV}" > "/tmp/srv_${TAG}.log" 2>&1 &
SPID=$!
log "server pid ${SPID}"

if ! wait_ready; then
    log "[ERROR] server not ready"
    kill ${SPID} 2>/dev/null
    exit 1
fi

log "=== bench 실행 ==="
./eval/bench.sh hybrid "${RUN_ENV}" || log "[WARN] bench rc=$?"

log "=== 서버 정리 ==="
kill ${SPID} 2>/dev/null
wait ${SPID} 2>/dev/null
pkill -f api_server 2>/dev/null || true
sleep 5

# ----------------------------------------------------------------------------
# path counter 요약
# ----------------------------------------------------------------------------
RESULT_DIR=$(ls -td /vllm_hybrid/eval/results/*_H_*_seqs2 2>/dev/null | head -1)
SERVER_LOG_BOOT="/tmp/srv_${TAG}.log"

log "=== [HYBRID-CPU-ATTN] path counter 요약 ==="
log "결과 dir    : ${RESULT_DIR}"
log "server log  : ${SERVER_LOG_BOOT}"

for f in "${RESULT_DIR}/hybrid_server_run.log" "${RESULT_DIR}/hybrid_server_boot.log" "${SERVER_LOG_BOOT}"; do
    [[ -f "${f}" ]] || continue
    echo "  --- from $(basename ${f}) ---"
    grep 'HYBRID-CPU-ATTN' "${f}" | tail -10 || echo "  (no [HYBRID-CPU-ATTN] in this log)"
    echo
done

log "=== 결론 찾는 법 ==="
cat <<'EOF'
  grep 출력의 `totals={'custom_avx': a, 'ipex': b, 'sdpa_batched': c, 'sdpa_loop': d}`
  비율로 해석:
    - sdpa_loop 가 dominant   → C 가설 (sdpa_loop fallback)
    - ipex 가 dominant + 느림 → A 가설 (IPEX long-ctx single-thread)
    - custom_avx 가 dominant  → 드물지만 custom_avx kernel 자체가 병렬 안 됨
  dominant path 확정 후 Phase 3 의 py-spy 로 stack 검증.
EOF
