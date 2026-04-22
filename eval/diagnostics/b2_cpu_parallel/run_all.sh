#!/usr/bin/env bash
# =============================================================================
# run_all.sh — B2 CPU Parallelism 검증 전체 orchestrator
#
# 한 번의 실행으로 Phase 1 + Phase 2 + Phase 3 를 순차 수행하고
# 결과를 단일 디렉토리에 모아 FINAL_REPORT.md 를 생성.
#
# 사용:
#   bash eval/diagnostics/b2_cpu_parallel/run_all.sh
#
# 선택 환경변수:
#   SKIP_PHASE2=1   — Phase 2 (재실행) 를 건너뛰고 Phase 1 + Phase 3 만
#   SKIP_PHASE3=1   — Phase 3 (live introspection) 생략
#   OUTPUT_LEN=32   — Phase 2 의 decode 길이 (기본 32)
#   PORT=8000       — server port
#
# 결과 저장 위치:
#   eval/diagnostics/b2_cpu_parallel/results/<YYYYMMDD_HHMMSS>/
#     FINAL_REPORT.md              ← 통합 보고서 (사람이 읽을 것)
#     phase1/
#       dispatch_static.txt        ← cpu_attn.py dispatch tree 추출
#     phase2/
#       server_boot.log            ← 서버 boot 로그
#       server_run.log             ← 서버 run 로그 (copy from eval/results)
#       hybrid.json                ← bench 결과 JSON (copy)
#       trace_counters.txt         ← [HYBRID-CPU-ATTN] 라인 추출
#       env_used.env               ← 사용한 env snapshot
#     phase3/
#       engine_<pid>_pyspy.txt     ← Python call stack
#       engine_<pid>_perf.txt      ← native hot function
#       engine_<pid>_threads.txt   ← thread state 분포
#       summary.md                 ← phase3 판정 가이드
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${SCRIPT_DIR}/results/${TS}"
PHASE1_DIR="${RESULTS_DIR}/phase1"
PHASE2_DIR="${RESULTS_DIR}/phase2"
PHASE3_DIR="${RESULTS_DIR}/phase3"
mkdir -p "${PHASE1_DIR}" "${PHASE2_DIR}" "${PHASE3_DIR}"

PORT="${PORT:-8000}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
SKIP_PHASE2="${SKIP_PHASE2:-0}"
SKIP_PHASE3="${SKIP_PHASE3:-0}"
READY_TIMEOUT=1200
PHASE3_WAIT=60  # Phase 2 bench 시작 후 몇 초 기다렸다 Phase 3 캡처

ENV_SRC="${SCRIPT_DIR}/g0_h100x8_qwen32b_longctx_trace.env"
RUN_ENV="/tmp/run_phase2.env"

log()  { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }
section() { echo; echo "════════════════════════════════════════════════════════════════"; log "$*"; echo "════════════════════════════════════════════════════════════════"; }

section "B2 CPU Parallelism 전체 검증 시작"
log "결과 저장 위치: ${RESULTS_DIR}"

# =============================================================================
# Phase 1 — 정적 코드 분석 (~5초)
# =============================================================================
section "Phase 1 — cpu_attn.py dispatch tree 정적 분석"
python3 "${SCRIPT_DIR}/phase1_dispatch_static.py" \
    > "${PHASE1_DIR}/dispatch_static.txt" 2>&1
log "  Phase 1 완료 → ${PHASE1_DIR}/dispatch_static.txt"

# =============================================================================
# Phase 2 — TRACE=1 짧은 run (~8-12분, server boot 포함)
# Phase 3 을 bench 실행 중간에 inject
# =============================================================================
if [[ "${SKIP_PHASE2}" == "1" ]]; then
    section "Phase 2 — SKIP (SKIP_PHASE2=1)"
else
    section "Phase 2 — VLLM_HYBRID_TRACE=1 단축 run 시작"

    # 기존 서버 정리
    pkill -f api_server 2>/dev/null || true
    pkill -f 'serve\.sh' 2>/dev/null || true
    sleep 3

    # env 준비 (OUTPUT_LEN override)
    cp "${ENV_SRC}" "${RUN_ENV}"
    sed -i "s/^OUTPUT_LEN=.*/OUTPUT_LEN=${OUTPUT_LEN}/" "${RUN_ENV}"
    cp "${RUN_ENV}" "${PHASE2_DIR}/env_used.env"
    log "  env (OUTPUT_LEN=${OUTPUT_LEN}) → ${PHASE2_DIR}/env_used.env"

    # 서버 기동
    BOOT_LOG="${PHASE2_DIR}/server_boot.log"
    log "  서버 기동 → ${BOOT_LOG}"
    ./eval/serve.sh hybrid "${RUN_ENV}" > "${BOOT_LOG}" 2>&1 &
    SPID=$!

    # ready 대기
    elapsed=0
    while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        if (( elapsed > READY_TIMEOUT )); then
            log "  [ERROR] ready timeout ${READY_TIMEOUT}s"
            kill "${SPID}" 2>/dev/null
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        (( elapsed % 30 == 0 )) && log "  waiting ready... (${elapsed}s)"
    done
    log "  서버 ready (${elapsed}s)"

    # bench 백그라운드 실행
    BENCH_LOG="${PHASE2_DIR}/bench.log"
    log "  bench 실행 (background) → ${BENCH_LOG}"
    ./eval/bench.sh hybrid "${RUN_ENV}" > "${BENCH_LOG}" 2>&1 &
    BENCH_PID=$!

    # =========================================================================
    # Phase 3 — bench 가 decode 중일 때 live introspection 주입
    # =========================================================================
    if [[ "${SKIP_PHASE3}" == "1" ]]; then
        log "  Phase 3 — SKIP (SKIP_PHASE3=1)"
    else
        log "  ${PHASE3_WAIT}초 대기 후 Phase 3 캡처 (bench 가 decode 안정화될 때까지)"
        sleep "${PHASE3_WAIT}"

        section "Phase 3 — stuck CPU engine live introspection"
        OUT_DIR="${PHASE3_DIR}" bash "${SCRIPT_DIR}/phase3_live_introspect.sh" \
            2>&1 | tee "${PHASE3_DIR}/capture.log"
        log "  Phase 3 완료 → ${PHASE3_DIR}/"
    fi

    # bench 완료 대기
    log "  bench 완료 대기..."
    wait "${BENCH_PID}" 2>/dev/null
    log "  bench 종료"

    # 서버 정리
    kill "${SPID}" 2>/dev/null
    wait "${SPID}" 2>/dev/null || true
    pkill -f api_server 2>/dev/null || true
    pkill -f 'serve\.sh' 2>/dev/null || true
    sleep 5

    # Phase 2 결과물 수집
    section "Phase 2 결과물 수집"
    # 가장 최근 H_C_*_seqs2 결과 디렉토리 찾아서 복사
    RECENT_RESULT=$(ls -td "${REPO_ROOT}/eval/results/"*_H_*_seqs2 2>/dev/null | head -1)
    if [[ -n "${RECENT_RESULT}" && -d "${RECENT_RESULT}" ]]; then
        log "  원본: ${RECENT_RESULT}"
        cp "${RECENT_RESULT}/hybrid_server_run.log" "${PHASE2_DIR}/server_run.log" 2>/dev/null || true
        cp "${RECENT_RESULT}/hybrid.json"            "${PHASE2_DIR}/hybrid.json"     2>/dev/null || true
        cp "${RECENT_RESULT}/applied_features.json"  "${PHASE2_DIR}/"                 2>/dev/null || true
        # [HYBRID-CPU-ATTN] counter 추출
        {
            echo "=== [HYBRID-CPU-ATTN] counter (from server run log) ==="
            grep 'HYBRID-CPU-ATTN' "${RECENT_RESULT}/hybrid_server_run.log" 2>/dev/null | tail -30 \
                || echo "(no [HYBRID-CPU-ATTN] markers)"
            echo
            echo "=== [HYBRID-CPU-ATTN-IPEX] counter (from server boot log) ==="
            grep 'HYBRID-CPU-ATTN-IPEX' "${RECENT_RESULT}/hybrid_server_boot.log" 2>/dev/null | tail -10 \
                || echo "(no [HYBRID-CPU-ATTN-IPEX] markers)"
            echo
            echo "=== [HYBRID-CPU-ATTN] counter (from server boot log — fallback) ==="
            grep 'HYBRID-CPU-ATTN' "${BOOT_LOG}" 2>/dev/null | tail -10 \
                || echo "(no [HYBRID-CPU-ATTN] in boot log)"
        } > "${PHASE2_DIR}/trace_counters.txt"
        log "  counter 추출 → ${PHASE2_DIR}/trace_counters.txt"
    else
        log "  [WARN] 최근 eval/results 에 seqs2 디렉토리 없음"
    fi
fi

# =============================================================================
# FINAL_REPORT.md 생성
# =============================================================================
section "FINAL_REPORT.md 생성"
REPORT="${RESULTS_DIR}/FINAL_REPORT.md"

{
    echo "# B2 CPU Parallelism 검증 통합 보고서"
    echo
    echo "- 실행 시각 (KST): ${TS}"
    echo "- 실행 브랜치: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    echo "- 실행 커밋: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "- 결과 디렉토리: \`${RESULTS_DIR#${REPO_ROOT}/}\`"
    echo
    echo "## 1. Phase 1 — 정적 dispatch 분석"
    echo
    echo "파일: [\`phase1/dispatch_static.txt\`](phase1/dispatch_static.txt)"
    echo
    echo "\`\`\`"
    grep -E 'Section A|Section B|\[.*\]  called|L [0-9]+ \|' "${PHASE1_DIR}/dispatch_static.txt" \
        2>/dev/null | head -40
    echo "\`\`\`"
    echo

    echo "## 2. Phase 3 — Live introspection"
    echo
    if [[ "${SKIP_PHASE3}" == "1" ]]; then
        echo "생략됨 (SKIP_PHASE3=1)"
    elif [[ -f "${PHASE3_DIR}/summary.md" ]]; then
        cat "${PHASE3_DIR}/summary.md"
        echo
        echo "### Thread 상태 요약"
        for f in "${PHASE3_DIR}"/engine_*_threads.txt; do
            [[ -f "${f}" ]] || continue
            echo
            echo "#### $(basename ${f})"
            echo "\`\`\`"
            head -30 "${f}"
            echo "\`\`\`"
        done
        echo "### py-spy 첫 head 20 lines"
        for f in "${PHASE3_DIR}"/engine_*_pyspy.txt; do
            [[ -f "${f}" ]] || continue
            echo "#### $(basename ${f})"
            echo "\`\`\`"
            head -30 "${f}" 2>/dev/null
            echo "\`\`\`"
        done
    else
        echo "Phase 3 결과 없음 (Phase 2 skip 또는 capture 실패)"
    fi
    echo

    echo "## 3. Phase 2 — TRACE counter 실측"
    echo
    if [[ "${SKIP_PHASE2}" == "1" ]]; then
        echo "생략됨 (SKIP_PHASE2=1)"
    elif [[ -f "${PHASE2_DIR}/trace_counters.txt" ]]; then
        echo "파일: [\`phase2/trace_counters.txt\`](phase2/trace_counters.txt)"
        echo
        echo "\`\`\`"
        cat "${PHASE2_DIR}/trace_counters.txt"
        echo "\`\`\`"
    else
        echo "Phase 2 결과 없음"
    fi
    echo

    echo "## 4. 판정 참고"
    echo
    cat <<'EOF'
| 증거 | 결론 |
|---|---|
| Phase 2 counter `sdpa_loop` dominant | **C** — dispatch 조건 수정 |
| Phase 2 counter `ipex` dominant 이고 여전히 느림 | **A** — IPEX 자체가 long-ctx 에서 single-thread |
| Phase 3 py-spy stack 이 Python attention 함수 | **B** — Python/GIL serialize |
| Phase 3 py-spy `<native>` + perf top `ipex_*_paged_attention` | **A** |
| Phase 3 py-spy `torch::sdpa` 경로 | **C** |
| Phase 3 threads.txt 대부분 S(sleep) + 소수 R | 스케줄 안 됨 → B 또는 native lock |

위 표와 실제 데이터 대조 → 가설 A/B/C 중 하나 선택 → `super_power/draft/B2/` 분석문서의 §8 레이어 3 / §11.1 B1 해석 갱신.
EOF
} > "${REPORT}"

section "완료"
log "FINAL_REPORT: ${REPORT}"
log "전체 결과: ${RESULTS_DIR}"
echo
ls -la "${RESULTS_DIR}"/
echo
log "보고서를 보려면: cat ${REPORT}"
