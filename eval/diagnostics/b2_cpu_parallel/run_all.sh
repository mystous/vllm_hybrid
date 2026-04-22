#!/usr/bin/env bash
# =============================================================================
# run_all.sh — B2 CPU Parallelism 검증 전체 orchestrator
#
# 한 번의 실행으로 Phase 1 + Phase 2 + Phase 3 를 순차 수행하고
# 결과를 단일 디렉토리에 모아 FINAL_REPORT.md 를 생성.
#
# 사용:
#   bash eval/diagnostics/b2_cpu_parallel/run_all.sh [OPTIONS]
#
# Options:
#   --env PATH            env file path (기본: g0_h100x8_qwen32b_longctx_trace.env)
#   --output-len N        OUTPUT_LEN override (기본: env 파일의 OUTPUT_LEN)
#   --port N              server port (기본 8000)
#   --ready-timeout N     server ready timeout seconds (기본 1200)
#   --phase3-wait N       bench 시작 후 phase3 까지 대기 초 (기본 60)
#   --perf-duration N     phase3 의 perf 샘플링 초 (기본 5)
#   --skip-phase2         Phase 2 (server + bench) skip
#   --skip-phase3         Phase 3 (live introspection) skip
#   --help                사용법 출력
#
# 모든 옵션은 CLI 인자로 전달. 이전의 환경변수 방식은 드물게 전파 안 되는
# 경우가 있어 제거. feature flag (예: HYBRID_CPU_ASYNC_EXECUTOR) 는
# env file 안에 직접 기입하는 게 정석.
#
# 결과 저장 위치:
#   eval/diagnostics/b2_cpu_parallel/results/<YYYYMMDD_HHMMSS>/
#     FINAL_REPORT.md              ← 통합 보고서 (사람이 읽을 것)
#     phase1/dispatch_static.txt   ← cpu_attn.py dispatch tree
#     phase2/
#       server_boot.log            ← 서버 boot 로그
#       server_run.log             ← 서버 run 로그 (copy from eval/results)
#       hybrid.json                ← bench 결과
#       trace_counters.txt         ← [HYBRID-CPU-ATTN] 추출
#       env_used.env               ← 사용한 env snapshot
#     phase3/
#       engine_<pid>_flame.svg     ← flame graph (SVG)
#       engine_<pid>_info.txt      ← ps/proc/OMP 정보
#       engine_<pid>_dump.txt      ← py-spy dump snapshot
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# 기본값
# ─────────────────────────────────────────────────────────────────────────────
ENV_SRC="${SCRIPT_DIR}/g0_h100x8_qwen32b_longctx_trace.env"
OUTPUT_LEN=""
PORT=8000
READY_TIMEOUT=1200
PHASE3_WAIT=60
PERF_DURATION=5
SKIP_PHASE2=0
SKIP_PHASE3=0

usage() {
    grep -E '^# ' "$0" | sed 's/^# \?//' | head -40
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI arg parsing
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)             ENV_SRC="$2"; shift 2 ;;
        --output-len)      OUTPUT_LEN="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --ready-timeout)   READY_TIMEOUT="$2"; shift 2 ;;
        --phase3-wait)     PHASE3_WAIT="$2"; shift 2 ;;
        --perf-duration)   PERF_DURATION="$2"; shift 2 ;;
        --skip-phase2)     SKIP_PHASE2=1; shift ;;
        --skip-phase3)     SKIP_PHASE3=1; shift ;;
        -h|--help)         usage ;;
        *) echo "[ERROR] unknown option: $1"; usage ;;
    esac
done

# env file 경로 resolve (상대 / 절대 모두 허용)
if [[ ! "${ENV_SRC}" = /* ]]; then
    # 상대 경로면 REPO_ROOT 기준
    if [[ -f "${REPO_ROOT}/${ENV_SRC}" ]]; then
        ENV_SRC="${REPO_ROOT}/${ENV_SRC}"
    elif [[ -f "${SCRIPT_DIR}/${ENV_SRC}" ]]; then
        ENV_SRC="${SCRIPT_DIR}/${ENV_SRC}"
    fi
fi
if [[ ! -f "${ENV_SRC}" ]]; then
    echo "[ERROR] env file not found: ${ENV_SRC}"
    exit 1
fi

# phase3 에서 참조하기 위해 export
export PERF_DURATION

TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${SCRIPT_DIR}/results/${TS}"
PHASE1_DIR="${RESULTS_DIR}/phase1"
PHASE2_DIR="${RESULTS_DIR}/phase2"
PHASE3_DIR="${RESULTS_DIR}/phase3"
mkdir -p "${PHASE1_DIR}" "${PHASE2_DIR}" "${PHASE3_DIR}"

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

    # env 준비 — ENV_SRC 복사. OUTPUT_LEN override 는 명시 시만.
    cp "${ENV_SRC}" "${RUN_ENV}"
    if [[ -n "${OUTPUT_LEN}" ]]; then
        sed -i "s/^OUTPUT_LEN=.*/OUTPUT_LEN=${OUTPUT_LEN}/" "${RUN_ENV}"
        log "  env OUTPUT_LEN override → ${OUTPUT_LEN}"
    fi
    cp "${RUN_ENV}" "${PHASE2_DIR}/env_used.env"
    log "  env file: ${ENV_SRC}"
    log "  snapshot: ${PHASE2_DIR}/env_used.env"

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

    # =========================================================================
    # Phase 3 캡처가 완료되면 bench 완료를 기다리지 않고 즉시 종료.
    # Phase 3 의 py-spy stack 이 핵심 증거이므로 이미 확보됐으면 충분.
    # 16K prefill on CPU 는 req 당 수십 분 걸려 bench 완료 기다리면 run_all 이
    # 1시간+ 걸림. 기존 "wait ${BENCH_PID}" 가 이 문제의 원인이었음.
    # =========================================================================
    log "  bench + 서버 즉시 종료 (Phase 3 증거 확보 완료)"
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
    kill -TERM "${SPID}"      2>/dev/null || true
    # bench 가 SIGTERM 에 반응 안 하면 3초 후 KILL
    sleep 3
    kill -9 "${BENCH_PID}" 2>/dev/null || true
    kill -9 "${SPID}"      2>/dev/null || true
    pkill -9 -f 'api_server|serve\.sh|benchmark_serving|CPU_EngineCore|GPU_EngineCore' 2>/dev/null || true
    sleep 2

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
    elif compgen -G "${PHASE3_DIR}/engine_*_flame.svg" > /dev/null \
      || compgen -G "${PHASE3_DIR}/engine_*_info.txt"  > /dev/null; then
        # 새 phase3 (flame.svg + info.txt + dump.txt)
        for SVG in "${PHASE3_DIR}"/engine_*_flame.svg; do
            [[ -f "${SVG}" ]] || continue
            EPID=$(basename "${SVG}" | sed -E 's/engine_([0-9]+)_flame\.svg/\1/')
            INFO="${PHASE3_DIR}/engine_${EPID}_info.txt"
            DUMP="${PHASE3_DIR}/engine_${EPID}_dump.txt"
            echo
            echo "### Engine PID ${EPID}"
            echo
            echo "- Flame graph: [\`phase3/engine_${EPID}_flame.svg\`](phase3/engine_${EPID}_flame.svg) (브라우저에서 열기)"
            [[ -f "${INFO}" ]] && echo "- Info: [\`phase3/engine_${EPID}_info.txt\`](phase3/engine_${EPID}_info.txt)"
            [[ -f "${DUMP}" ]] && echo "- Dump: [\`phase3/engine_${EPID}_dump.txt\`](phase3/engine_${EPID}_dump.txt)"
            echo

            # Info 요약 — ps top 10 + OMP libs + thread 이름 분포
            if [[ -f "${INFO}" ]]; then
                echo "#### Process 요약"
                echo "\`\`\`"
                # 첫 3줄 (PID / threads / cpus) + OMP + Thread 이름 + ps top-10
                sed -n '1,3p' "${INFO}"
                echo
                awk '/### OMP\/BLAS/,/^$/' "${INFO}" | head -10
                awk '/### Thread 이름 분포/,/^$/' "${INFO}" | head -10
                awk '/### ps -L top-10/,/^$/' "${INFO}" | head -15
                echo "\`\`\`"
                echo
            fi

            # Flame graph 에서 top hot functions 추출 (SVG <title> element)
            echo "#### Top hot functions (flame graph 샘플 상위)"
            echo "\`\`\`"
            grep -oE '<title>[^<]+\([0-9]+ samples,[^<]+</title>' "${SVG}" \
                | sed -E 's/<\/?title>//g' \
                | python3 -c "
import sys, re
rows = []
for line in sys.stdin:
    m = re.search(r'\((\d+) samples', line)
    if m:
        rows.append((int(m.group(1)), line.strip()))
rows.sort(key=lambda r: -r[0])
for n, line in rows[:20]:
    print(f'{n:5d}  {line}')
" 2>/dev/null || echo "(flame graph 파싱 실패)"
            echo "\`\`\`"
            echo
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
log ""
log "raw data 도 git 에 포함하려면:"
log "  git add eval/diagnostics/b2_cpu_parallel/results/${TS}/"
log "  git commit -m \"diag(b2): ${TS} run 결과\" && git push"
