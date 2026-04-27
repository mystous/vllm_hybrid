#!/usr/bin/env bash
# =============================================================================
# run_prod_cold_verify.sh — IDE_006 cold-path 실 발화 검증용 short verifier.
#
# simd_verify (--max-prompts 8) 는 GPU KV pool 의 ~10% 만 차지해 cold-tier
# eviction 이 트리거되지 않는다. 즉 그 wrapper 의 e2e_quick 은 hot path
# 만 도는 케이스라 IDE_006 의 본 기능 (cold path firing + per-seq 필터
# + AMX kernel 실행) 검증으로는 부족하다.
#
# 본 wrapper 는 NUM_PROMPTS 를 GPU KV pool 한계 위 (default 100) 로
# 두어 실제 cold KV 가 형성되고 hot_cold_attention 의 cold path 가
# fire 되는지를 확인한다. 다만 OUTPUT_LEN 은 짧게 (default 16) 해서
# 풀 prod_smoke (NUM_PROMPTS=100, OUTPUT_LEN=128) 보다 회전이 빨라
# 회귀 검증 / fix 후 단발 회차로 적합하다.
#
# 산출물:
#   eval/results/<TS>_<HW_TAG>_cold_verify/
#     ├── README.md                   meta + 요약
#     ├── isa_info.txt                CPU/GPU/NUMA 스냅샷
#     ├── e2e.log / e2e.stderr.log    run_e2e_accuracy.py 로그
#     ├── e2e_artifacts/split_on.json 생성 결과 + scheduler 통계
#     ├── monitor_cpu.csv             1 초 간격 CPU 사용률 시계열
#     ├── monitor_gpu.csv             1 초 간격 GPU 사용률 시계열
#     └── monitor.log                 monitor.py stdout
#
# 통과 기준:
#   1. e2e RC = 0 (engine 안 죽음, 100/100 success)
#   2. e2e.stderr.log 에 'ColdPath fired' 또는 IDE_006 dispatcher 발화
#      라인 1 회 이상 (cold KV 가 실제로 만들어졌는지)
#   3. monitor_cpu.csv 의 평균 CPU 사용률 baseline 대비 의미있게 상승
#      (IDE_006 의 본 가치 — CPU 활용 — 정량 신호)
#
# Usage:
#   bash eval/run_prod_cold_verify.sh                # 기본 (100/16) + 결과 push 안 함
#   bash eval/run_prod_cold_verify.sh --push         # 결과 자동 commit + push
#   NUM_PROMPTS=80 bash eval/run_prod_cold_verify.sh # KV pool 경계 직전
#   OUTPUT_LEN=8 bash eval/run_prod_cold_verify.sh   # 더 짧게
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_cold_verify"
mkdir -p "${OUT_DIR}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python)"
fi

PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

# Defaults — operator can override via env.
NUM_PROMPTS="${NUM_PROMPTS:-100}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --------------------------------------------------------------------- meta

log "writing meta to ${OUT_DIR}/README.md"
{
    echo "# ${TS}_${HW_TAG}_cold_verify"
    echo
    echo "- timestamp: ${TS}"
    echo "- hw_tag:    ${HW_TAG}"
    echo "- branch:    $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit:    $(git rev-parse HEAD)"
    echo "- python:    ${PYTHON}"
    echo "- vllm:      $(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
    echo "- NUM_PROMPTS: ${NUM_PROMPTS}"
    echo "- OUTPUT_LEN:  ${OUTPUT_LEN}"
    echo "- MONITOR_INTERVAL: ${MONITOR_INTERVAL}s"
    echo
    echo "## components"
    echo "- eval/run_e2e_accuracy.py --split-on-only — IDE_006 cold path 실 발화"
    echo "- eval/monitor.py — CPU/GPU 사용률 시계열 캡처 (interval=${MONITOR_INTERVAL}s)"
    echo
    echo "## 통과 기준"
    echo "- e2e RC = 0 (engine 안 죽음)"
    echo "- e2e_artifacts/split_on.json 의 num_completed = ${NUM_PROMPTS}"
    echo "- e2e.stderr.log 에 cold path 발화 흔적 (IDE_006 dispatcher / AMX trace)"
    echo "- monitor_cpu.csv 평균 CPU 사용률이 baseline 대비 의미있는 수준"
} > "${OUT_DIR}/README.md"

log "capturing CPU/GPU/NUMA snapshot to ${OUT_DIR}/isa_info.txt"
{
    echo "=== /proc/cpuinfo (ISA flags) ==="
    grep -oE 'avx512[a-z_0-9]*|amx_[a-z_]*|avx2|sse4_2' /proc/cpuinfo | sort -u
    echo
    echo "=== uname -r ==="
    uname -r
    echo
    echo "=== ulimit -l ==="
    ulimit -l
    echo
    echo "=== free -h ==="
    free -h
    echo
    echo "=== nvidia-smi (snapshot) ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>&1 || echo "nvidia-smi unavailable"
    echo
    echo "=== numactl --hardware ==="
    numactl --hardware 2>&1 || echo "numactl unavailable"
} > "${OUT_DIR}/isa_info.txt"

# --------------------------------------------------------------------- monitor

log "starting monitor (interval=${MONITOR_INTERVAL}s)"
"${PYTHON}" "${SCRIPT_DIR}/monitor.py" "${OUT_DIR}/monitor" \
    --interval "${MONITOR_INTERVAL}" >"${OUT_DIR}/monitor.log" 2>&1 &
MONITOR_PID=$!
log "monitor PID: ${MONITOR_PID}"

cleanup() {
    if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        log "stopping monitor (PID=${MONITOR_PID})"
        kill -INT "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------- e2e

log "[1/1] e2e accuracy (split-on-only, NUM_PROMPTS=${NUM_PROMPTS}, OUTPUT_LEN=${OUTPUT_LEN})"
E2E_RC=0
HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/vllm_original_long_ctx.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_long_ctx.env" \
    --split-on-only \
    --max-prompts "${NUM_PROMPTS}" \
    --max-tokens "${OUTPUT_LEN}" \
    --logprobs 0 \
    --output-dir "${OUT_DIR}/e2e_artifacts" \
    > >(tee "${OUT_DIR}/e2e.log") 2> >(tee "${OUT_DIR}/e2e.stderr.log" >&2) || E2E_RC=$?
log "  e2e exit=${E2E_RC}"

# --------------------------------------------------------------------- summary

# 발화 흔적 카운트.
COLD_FIRINGS=$(grep -cE "IDE_006 .* dispatcher|forward_partial_with_lse|hot_cold_attention" \
    "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)
AMX_TRACE=$(grep -cE "AMX trace|\[IDE_006/TSK_004 profile" \
    "${OUT_DIR}/e2e.stderr.log" 2>/dev/null || echo 0)
COMPLETED_LINE=$(grep -E "batched generate complete:|done in" "${OUT_DIR}/e2e.log" | tail -2)

# 평균 CPU 사용률 (psutil 의 cpu_percent 평균).
CPU_AVG=$(awk -F, 'NR>1 && $2!="" {sum+=$2; n++} END{if(n>0) printf "%.1f", sum/n; else print "(no samples)"}' \
    "${OUT_DIR}/monitor_cpu.csv" 2>/dev/null || echo "(no monitor_cpu.csv)")
GPU_AVG=$(awk -F, 'NR>1 && $3!="" {sum+=$3; n++} END{if(n>0) printf "%.1f", sum/n; else print "(no samples)"}' \
    "${OUT_DIR}/monitor_gpu.csv" 2>/dev/null || echo "(no monitor_gpu.csv)")

{
    echo
    echo "## exit codes"
    echo "- e2e:                       ${E2E_RC}"
    echo
    echo "## 발화 신호"
    echo "- IDE_006 dispatcher 라인:    ${COLD_FIRINGS} 회"
    echo "- AMX trace / partial profile: ${AMX_TRACE} 회"
    echo "- bench 완료 라인:"
    echo "${COMPLETED_LINE}" | sed 's/^/    /'
    echo
    echo "## 사용률 (시계열 평균)"
    echo "- CPU 사용률 평균: ${CPU_AVG} %"
    echo "- GPU 사용률 평균: ${GPU_AVG} %"
    echo
    echo "## 다음 작업"
    echo "- 발화 신호 = 0 이면 cold path 안 탔음 → NUM_PROMPTS 더 키워야 함"
    echo "- 발화 신호 > 0 + e2e RC = 0 이면 검증 통과"
    echo "- CPU 평균이 baseline 대비 거의 같으면 overlap 미작동 → §4.6 작업 필요"
} >> "${OUT_DIR}/README.md"

log "artifacts -> ${OUT_DIR}"
log "key signals:"
log "  - IDE_006 dispatcher firings: ${COLD_FIRINGS}"
log "  - AMX trace / profile lines : ${AMX_TRACE}"
log "  - CPU avg util              : ${CPU_AVG}%"
log "  - GPU avg util              : ${GPU_AVG}%"

# --------------------------------------------------------------------- push

if [[ ${PUSH} -eq 1 ]]; then
    log "staging eval/results/ + git push"
    git add eval/results/
    if git diff --cached --quiet; then
        log "  nothing to commit"
    else
        git commit -m "chore(eval): IDE_006 cold-path verify ${TS} @ ${HW_TAG} (${NUM_PROMPTS}x${OUTPUT_LEN})"
        BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        git push origin "${BRANCH}"
    fi
fi

[[ ${E2E_RC} -eq 0 ]] && exit 0 || exit 1
