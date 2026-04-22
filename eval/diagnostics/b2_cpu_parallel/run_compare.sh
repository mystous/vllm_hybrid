#!/usr/bin/env bash
# =============================================================================
# run_compare.sh — X Phase 4/5 — sync vs async 성능 비교
#
# 목적: HYBRID_CPU_ASYNC_EXECUTOR=0 (sync baseline) 과 =1 (X Phase 3 async)
#       를 **같은 workload 로 연속 실행**하고 bench 완주 후 duration /
#       throughput / completed 차이를 자동 비교. FINAL_REPORT 생성.
#
# run_all.sh 와 차이:
#   - phase3 snapshot 없음 (bench 완주 기다림)
#   - 두 번 실행 (sync + async) 자동 비교
#   - flame graph 수집 안 함 (성능 숫자가 목표)
#
# 사용:
#   bash eval/diagnostics/b2_cpu_parallel/run_compare.sh [OPTIONS]
#
# Options:
#   --env PATH            env file (default: g0_h100x8_qwen32b_light_trace.env)
#   --port N              server port (default 8000)
#   --ready-timeout N     server ready timeout (default 1200)
#   --bench-timeout N     bench max duration (default 7200 = 2h)
#   --skip-sync           sync 생략, async 만 실행
#   --skip-async          async 생략, sync 만 실행
#   --help
#
# 결과: eval/diagnostics/b2_cpu_parallel/results/compare_<ts>/
#   sync/
#     server_boot.log, bench.log, hybrid.json, env_used.env
#   async/
#     (동일)
#   COMPARE_REPORT.md           ← 비교 테이블 + 판정
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# 기본값
ENV_SRC="${SCRIPT_DIR}/g0_h100x8_qwen32b_light_trace.env"
PORT=8000
READY_TIMEOUT=1200
BENCH_TIMEOUT=7200
SKIP_SYNC=0
SKIP_ASYNC=0

usage() { grep -E '^# ' "$0" | sed 's/^# \?//' | head -35; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)             ENV_SRC="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --ready-timeout)   READY_TIMEOUT="$2"; shift 2 ;;
        --bench-timeout)   BENCH_TIMEOUT="$2"; shift 2 ;;
        --skip-sync)       SKIP_SYNC=1; shift ;;
        --skip-async)      SKIP_ASYNC=1; shift ;;
        -h|--help)         usage ;;
        *) echo "[ERROR] unknown: $1"; usage ;;
    esac
done

if [[ ! "${ENV_SRC}" = /* ]]; then
    [[ -f "${REPO_ROOT}/${ENV_SRC}" ]] && ENV_SRC="${REPO_ROOT}/${ENV_SRC}"
fi
[[ -f "${ENV_SRC}" ]] || { echo "[ERROR] env not found: ${ENV_SRC}"; exit 1; }

TS=$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')
RESULTS_DIR="${SCRIPT_DIR}/results/compare_${TS}"
mkdir -p "${RESULTS_DIR}/sync" "${RESULTS_DIR}/async"

log()     { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }
section() { echo; echo "════════════════════════════════════════════════════════════════"; log "$*"; echo "════════════════════════════════════════════════════════════════"; }

# ─────────────────────────────────────────────────────────────────────────────
# 철저한 server cleanup — GPU 메모리 + port + 잔여 프로세스까지 기다림
# ─────────────────────────────────────────────────────────────────────────────
cleanup_servers() {
    # 1. 관련 프로세스 3 pass pkill (일부가 재생성되는 경우 대비)
    for pass in 1 2 3; do
        pkill -9 -f 'api_server|serve\.sh|bench\.sh|benchmark_serving|CPU_EngineCore|GPU_EngineCore|VllmWorker|vllm\.entrypoints|multiproc_executor' 2>/dev/null || true
        sleep 2
    done

    # 2. Port ${PORT} TIME_WAIT 해제 대기 (최대 60s)
    local waited=0
    while ss -tln 2>/dev/null | grep -q ":${PORT} " && (( waited < 60 )); do
        sleep 2; waited=$((waited + 2))
    done
    (( waited > 0 )) && log "  port ${PORT} 해제 대기 ${waited}s"

    # 3. GPU memory 해제 대기 — 모든 GPU 가 <1GB 사용 상태가 되어야 (최대 60s)
    if command -v nvidia-smi >/dev/null 2>&1; then
        waited=0
        while (( waited < 60 )); do
            local max_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
            [[ -z "${max_mb}" ]] && break
            (( max_mb < 1000 )) && break
            sleep 3; waited=$((waited + 3))
        done
        (( waited > 0 )) && log "  GPU memory 해제 대기 ${waited}s"
    fi

    sleep 3
}

# ─────────────────────────────────────────────────────────────────────────────
# 공통: 한 번의 run 을 수행 (mode=sync 또는 async)
# ─────────────────────────────────────────────────────────────────────────────
run_one() {
    local MODE="$1"      # "sync" or "async"
    local OUT_DIR="${RESULTS_DIR}/${MODE}"
    local RUN_ENV="/tmp/run_compare_${MODE}.env"
    local FLAG_VAL=$([[ "${MODE}" == "async" ]] && echo 1 || echo 0)

    section "[${MODE}] 시작 (HYBRID_CPU_ASYNC_EXECUTOR=${FLAG_VAL})"

    # env 파일 준비 — flag override
    cp "${ENV_SRC}" "${RUN_ENV}"
    # HYBRID_CPU_ASYNC_EXECUTOR 라인이 있으면 교체, 없으면 추가
    if grep -q '^HYBRID_CPU_ASYNC_EXECUTOR=' "${RUN_ENV}"; then
        sed -i "s/^HYBRID_CPU_ASYNC_EXECUTOR=.*/HYBRID_CPU_ASYNC_EXECUTOR=${FLAG_VAL}/" "${RUN_ENV}"
    else
        echo "HYBRID_CPU_ASYNC_EXECUTOR=${FLAG_VAL}" >> "${RUN_ENV}"
    fi
    cp "${RUN_ENV}" "${OUT_DIR}/env_used.env"

    # 이전 서버 정리 — 철저히 (GPU 메모리 + port 포함)
    log "  이전 서버 cleanup 중..."
    cleanup_servers

    # 서버 기동
    local BOOT_LOG="${OUT_DIR}/server_boot.log"
    log "  서버 기동 → ${BOOT_LOG}"
    ./eval/serve.sh hybrid "${RUN_ENV}" > "${BOOT_LOG}" 2>&1 &
    local SPID=$!

    # ready 대기
    local elapsed=0
    while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        if (( elapsed > READY_TIMEOUT )); then
            log "  [ERROR] ready timeout ${READY_TIMEOUT}s"
            kill "${SPID}" 2>/dev/null; return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        (( elapsed % 30 == 0 )) && log "  waiting ready... (${elapsed}s)"
    done
    log "  ready (${elapsed}s)"

    # bench 완주 (phase3 없음)
    local BENCH_LOG="${OUT_DIR}/bench.log"
    local T0=$(date +%s)
    log "  bench 실행 → ${BENCH_LOG}"
    timeout --kill-after=30 "${BENCH_TIMEOUT}" \
        ./eval/bench.sh hybrid "${RUN_ENV}" > "${BENCH_LOG}" 2>&1 || {
        local RC=$?
        log "  [WARN] bench rc=${RC} (timeout 가능)"
    }
    local T1=$(date +%s)
    local BENCH_WALL=$((T1 - T0))
    log "  bench 완료 (wall=${BENCH_WALL}s)"

    # 서버 정리 — 철저히
    kill -TERM "${SPID}" 2>/dev/null || true
    sleep 3
    log "  [${MODE}] 서버 cleanup 중..."
    cleanup_servers

    # 결과 파일 복사 — eval/results/ 의 최신 H_C_*_seqs* 디렉토리
    local RECENT=$(ls -td "${REPO_ROOT}/eval/results/"*_H_C_*_seqs* 2>/dev/null | head -1)
    if [[ -n "${RECENT}" && -d "${RECENT}" ]]; then
        log "  결과 copy: ${RECENT}"
        cp "${RECENT}"/hybrid.json          "${OUT_DIR}/"  2>/dev/null || true
        cp "${RECENT}"/applied_features.json "${OUT_DIR}/"  2>/dev/null || true
        cp "${RECENT}"/system_info.json     "${OUT_DIR}/"  2>/dev/null || true
    fi

    # wall time 기록
    echo "${BENCH_WALL}" > "${OUT_DIR}/bench_wall_seconds.txt"
    log "  [${MODE}] 완료"
}

# ─────────────────────────────────────────────────────────────────────────────
# 메인 — sync, async 순차 실행
# ─────────────────────────────────────────────────────────────────────────────
section "X Phase 4/5 — sync vs async 비교 시작"
log "env      : ${ENV_SRC}"
log "result   : ${RESULTS_DIR}"

# 스크립트 시작 직전 — 이전 실행 잔여 프로세스 / GPU 메모리 / port 까지 정리
log "pre-run cleanup..."
cleanup_servers

[[ "${SKIP_SYNC}"  == "0" ]] && run_one sync
[[ "${SKIP_ASYNC}" == "0" ]] && run_one async

# 스크립트 종료 직전 — 다음 수동 실행이 깨끗한 상태에서 시작되도록 최종 정리
log "post-run cleanup..."
cleanup_servers

# ─────────────────────────────────────────────────────────────────────────────
# 비교 report
# ─────────────────────────────────────────────────────────────────────────────
section "COMPARE_REPORT 생성"

REPORT="${RESULTS_DIR}/COMPARE_REPORT.md"
python3 - "${RESULTS_DIR}" "${REPORT}" << 'PYTHON_EOF'
import json, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
report_path = Path(sys.argv[2])

def load(mode):
    d = {}
    hj = results_dir / mode / "hybrid.json"
    if hj.exists():
        try: d.update(json.loads(hj.read_text()))
        except Exception as e: d["_load_err"] = str(e)
    wf = results_dir / mode / "bench_wall_seconds.txt"
    if wf.exists():
        try: d["bench_wall_seconds"] = int(wf.read_text().strip())
        except: pass
    return d

sync = load("sync")
asyn = load("async")

def g(d, k, default=None):
    return d.get(k, default) if d else default

def pct(a, b):
    if a is None or b is None: return "—"
    if b == 0: return "—"
    return f"{(b - a) / a * 100:+.1f}%"

lines = [
    "# X Phase 4/5 — Sync vs Async 비교 Report",
    "",
    f"결과 디렉토리: `{results_dir.name}`",
    "",
    "## 핵심 metric",
    "",
    "| 항목 | sync (HYBRID_CPU_ASYNC_EXECUTOR=0) | async (=1) | Δ |",
    "|---|---:|---:|---:|",
]

metrics = [
    ("completed",           "completed",           None),
    ("total_output_tokens", "total_output_tokens", None),
    ("duration (bench, s)", "duration",            "lower=better"),
    ("bench wall (s)",      "bench_wall_seconds",  "lower=better"),
    ("request_throughput",  "request_throughput",  "higher=better"),
    ("output_throughput",   "output_throughput",   "higher=better"),
    ("mean_ttft_ms",        "mean_ttft_ms",        "lower=better"),
    ("p99_ttft_ms",         "p99_ttft_ms",         "lower=better"),
    ("mean_tpot_ms",        "mean_tpot_ms",        "lower=better"),
    ("p99_tpot_ms",         "p99_tpot_ms",         "lower=better"),
]

for label, key, _ in metrics:
    sv, av = g(sync, key), g(asyn, key)
    fmt = (lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)) if sv is not None or av is not None else (lambda v: "—")
    svs = fmt(sv) if sv is not None else "—"
    avs = fmt(av) if av is not None else "—"
    dlt = pct(sv, av) if isinstance(sv, (int, float)) and isinstance(av, (int, float)) else "—"
    lines.append(f"| {label} | {svs} | {avs} | {dlt} |")

# 판정
lines += [
    "",
    "## 판정",
    "",
]

if not sync.get("completed") and not asyn.get("completed"):
    lines.append("- 데이터 부족 — sync/async 둘 다 bench 완료 결과 없음")
elif sync.get("completed") != asyn.get("completed"):
    lines.append(f"- ⚠ completed 수가 다름 (sync={sync.get('completed')}, async={asyn.get('completed')}) — correctness 확인 필요")
else:
    d_sync = sync.get("duration")
    d_async = asyn.get("duration")
    if d_sync and d_async:
        improv = (d_sync - d_async) / d_sync * 100
        if improv > 10:
            lines.append(f"- ✅ async 가 sync 대비 **{improv:.1f}% 빠름** (duration 기준) — X pipeline 의미 있는 이득")
        elif improv > 0:
            lines.append(f"- 🟡 async 가 sync 대비 {improv:.1f}% 빠름 — marginal, Phase 1 보수 추정치 (5~15%) 범위")
        else:
            lines.append(f"- ❌ async 가 sync 대비 {-improv:.1f}% 느림 — overlap 이 thread overhead 미만")
    else:
        lines.append("- duration 값 부족 — 비교 불가")

lines += [
    "",
    "## 데이터 아티팩트",
    "",
    "- `sync/` — sync baseline 전체 결과",
    "- `async/` — async 실행 결과",
    "- 각 디렉토리의 `hybrid.json` / `bench.log` / `server_boot.log` / `env_used.env`",
]

report_path.write_text("\n".join(lines))
print(f"report saved: {report_path}")
PYTHON_EOF

log "COMPARE_REPORT: ${REPORT}"
echo
cat "${REPORT}"
