#!/usr/bin/env bash
# =============================================================================
# run_prod_tsk009_validation.sh — TSK_009 사용자 framing fix 의 통합 검증.
#
# 사용자 framing (2026-04-29):
#   1. CPU 가 GPU 의 작업을 절대 방해하면 안 됨
#   2. Cold Tier 가 host 도착 순간부터 partial attention 진행
#   3. partial attention 이 GPU 가 필요할 때까지 준비 안 되면 폐기
#   4. 폐기 시 GPU 가 스스로 계산
#
# 두 invariant 검증:
#   invariant 1 — 속도: cold-tier ON IDE_006 ON 이 cold-tier ON IDE_006 OFF
#                 baseline 보다 *느려지면 안 됨* (= IDE_006 추가 cost ≤ 0)
#   invariant 2 — 활용: CPU 작업이 *모두* 버려지지 않음. _record_cold_outcome
#                 의 merged 횟수 (= hot subset paged FA + merge 호출 수) > 0
#
# 3 시나리오 (input + output = 16384, max_model_len 한도):
#   - input_heavy:  INPUT=15360, OUTPUT=1024
#   - output_heavy: INPUT=1024,  OUTPUT=15360
#   - equal:        INPUT=8192,  OUTPUT=8192
# 각 시나리오 100 prompts.
#
# 각 시나리오에서 두 회차:
#   - mode=B: cold-tier ON, IDE_006 OFF (`ide006_cold_kv_long_ctx.env`) — 비교 baseline
#   - mode=C: cold-tier ON, IDE_006 ON  (`ide006_cold_kv_split_on_long_ctx.env`) — fix 적용
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_hwtag.sh"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"

TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_tsk009_validation"
mkdir -p "${OUT_DIR}"

# IDE_006 partition path 의 deadline 활성화. fix 의 non-blocking poll 은
# deadline > 0 일 때만 진입. 100ms 는 이전 값 유지 (poll 자체는 즉시 검사라
# deadline 실제 wait 는 안 일어남).
export VLLM_COLD_KV_FALLBACK_DEADLINE_MS="${VLLM_COLD_KV_FALLBACK_DEADLINE_MS:-100}"
export VLLM_COLD_KV_OUTCOME_LOG_LIMIT="${VLLM_COLD_KV_OUTCOME_LOG_LIMIT:-50}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

{
    echo "# ${TS}_${HW_TAG}_tsk009_validation"
    echo "- num_prompts: ${NUM_PROMPTS}"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "- commit: $(git rev-parse HEAD)"
    echo
    echo "## invariants"
    echo "- 1: B (cold-tier ON, IDE_006 OFF) wall-time vs C (IDE_006 ON, fix) — C ≤ B"
    echo "- 2: C 회차의 cold-outcome merged > 0 (CPU 작업 활용된 layer 수)"
} > "${OUT_DIR}/README.md"

_run() {
    local label="$1" mode="$2" input_len="$3" output_len="$4"
    local SCEN_DIR="${OUT_DIR}/${label}_${mode}"
    mkdir -p "${SCEN_DIR}/e2e_artifacts"

    log "==> scenario=${label} mode=${mode} (input=${input_len} output=${output_len})"

    local TMP_BASE="${SCEN_DIR}/baseline.env"
    local TMP_SPLIT="${SCEN_DIR}/split_on.env"
    local SRC_SPLIT EXTRA_FLAG
    if [[ "${mode}" == "B" ]]; then
        SRC_SPLIT="${SCRIPT_DIR}/envs/ide006_cold_kv_long_ctx.env"
        EXTRA_FLAG="--allow-equivalent-config"
    else
        SRC_SPLIT="${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_long_ctx.env"
        EXTRA_FLAG=""
    fi

    sed -E -e "s|^INPUT_LEN=.*|INPUT_LEN=${input_len}|" \
           -e "s|^OUTPUT_LEN=.*|OUTPUT_LEN=${output_len}|" \
           -e "s|^NUM_PROMPTS=.*|NUM_PROMPTS=${NUM_PROMPTS}|" \
           "${SCRIPT_DIR}/envs/vllm_original_long_ctx.env" > "${TMP_BASE}"
    sed -E -e "s|^INPUT_LEN=.*|INPUT_LEN=${input_len}|" \
           -e "s|^OUTPUT_LEN=.*|OUTPUT_LEN=${output_len}|" \
           -e "s|^NUM_PROMPTS=.*|NUM_PROMPTS=${NUM_PROMPTS}|" \
           "${SRC_SPLIT}" > "${TMP_SPLIT}"

    local RC=0
    HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
        --baseline-env "${TMP_BASE}" \
        --split-on-env "${TMP_SPLIT}" \
        --max-prompts "${NUM_PROMPTS}" \
        --max-tokens "${output_len}" \
        --logprobs 0 \
        ${EXTRA_FLAG} \
        --output-dir "${SCEN_DIR}/e2e_artifacts" \
        > >(tee "${SCEN_DIR}/e2e.log") 2> >(tee "${SCEN_DIR}/e2e.stderr.log" >&2) || RC=$?

    local OUTCOMES=""
    if [[ "${mode}" == "C" ]]; then
        OUTCOMES=$(grep "TSK_009 cold-outcome" "${SCEN_DIR}/e2e.stderr.log" 2>/dev/null | tail -3)
    fi
    {
        echo
        echo "## ${label}_${mode} (input=${input_len}, output=${output_len})"
        echo "- exit: ${RC}"
        echo '```'
        grep -E "\[baseline\]|\[split_on\]|done in" "${SCEN_DIR}/e2e.log" | head -10
        if [[ -n "${OUTCOMES}" ]]; then
            echo
            echo "### cold-outcome (last 3)"
            echo "${OUTCOMES}"
        fi
        echo '```'
    } >> "${OUT_DIR}/README.md"

    log "    ${label}_${mode} exit=${RC}"
    sleep 5
}

# Each scenario: B then C
_run input_heavy  B 15360 1024
_run input_heavy  C 15360 1024
_run output_heavy B 1024  15360
_run output_heavy C 1024  15360
_run equal        B 8192  8192
_run equal        C 8192  8192

log "all scenarios complete. results -> ${OUT_DIR}"
