#!/usr/bin/env bash
# =============================================================================
# run_prod_tsk011_deadline_sweep.sh — TSK_011 deadline 별 TST_003 sweep.
#
# 목적: VLLM_COLD_KV_FALLBACK_DEADLINE_MS 가 fallback 발동률 / D-ii 봉합 / 처
#   리량 에 어떤 영향을 주는지 다중 회차로 sweep. 100ms 는 별도 회차로 이미
#   진행됨 (fallback 100%) — 본 sweep 은 1000ms / 5000ms 두 점 추가하여 3 점
#   분포로 비교.
#
# 회차 구성: 각 deadline 마다 run_prod_quick_tst003.sh 1 회 (max-prompts=30 /
#   max-tokens=16 / logprobs=1, baseline + split_on 70B+TP=8).
#
# 실행 시간 견적: ~24~30 분 (12~15 분 × 2 deadline).
#
# 산출물: 각 deadline 마다 별도 디렉토리. 결과 비교는 마지막에 sweep 요약.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Default sweep — 1000 ms, 5000 ms. 100 ms 결과는 별도 회차에 이미 존재.
DEADLINES="${DEADLINES:-1000 5000}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

DIRS=()
for ms in $DEADLINES; do
    log "==> deadline=${ms}ms"
    VLLM_COLD_KV_FALLBACK_DEADLINE_MS="${ms}" \
        bash "${SCRIPT_DIR}/run_prod_quick_tst003.sh" || {
        log "회차 실패 (deadline=${ms}ms) — 이후 회차 skip"
        break
    }
    NEW=$(ls -td "${SCRIPT_DIR}/results/"*_quick_tst003 | head -1)
    DIRS+=("${ms}:${NEW}")
    log "    → ${NEW}"
    # 다음 회차 진입 전 잠깐 — vllm engine shutdown 깨끗하게.
    sleep 5
done

log "sweep complete. result dirs:"
for d in "${DIRS[@]}"; do log "  ${d}"; done
