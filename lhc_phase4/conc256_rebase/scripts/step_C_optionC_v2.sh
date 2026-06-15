#!/usr/bin/env bash
# Step C — Adaptive regime gate (PREFIX_HOT-aware Option C v2).
#
# For 6 workloads × {vanilla, optionC_v2}, 3 sweeps each.
#
#   vanilla       : VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
#   optionC_v2    : adds VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1
#                   VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1
#                   VLLM_LHC_AMX_C3_LIB=... (Path 1 routed via regime gate —
#                   only activates when detector says PREFIX_HOT)
#
# Goal:  on code workload → PREFIX_HOT → Path 1 ON → match step3 +10.67%.
#        on sonnet/chat/balanced → GPU_SATURATED → Path 1 OFF → vanilla parity.

set -uo pipefail
HERE="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/scripts"
source "${HERE}/lib_common.sh"

export PORT=8001
export NPROMPTS=500
export INPUT_LEN=1024
export OUTPUT_LEN=2048
export CONC=256

OUTDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/optionC_v2_runs"
LOGDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/optionC_v2_logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

BW_PY="/workspace/host_vllm_hybrid/vllm_config_perf/gating/benchmark_workloads.py"
PY="/workspace/vllm_dev_prj/bin/python3"
LIB="/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"

# Default to all 6 workloads — but no PREFIX_HOT-specific filtering;
# the gate is supposed to self-disable on non-PREFIX_HOT workloads.
WORKLOADS=${WORKLOADS:-"sonnet chat code balanced sonnet-heavy code-heavy"}
SWEEPS=${SWEEPS:-"1 2 3"}

CONFIGS=(
    "vbw|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1"
    "optCv2|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_AMX_C3_LIB=${LIB} VLLM_LHC_REGIME_ADAPTIVE=1"
)

boot_vllm_port() {
    local TAG=$1 ENV_PRE=$2 LOG=$3
    cleanup_orphans
    eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        setsid ${VLLM_BIN} serve ${MODEL} \
          --port ${PORT} --host 127.0.0.1 \
          --tensor-parallel-size ${TP} \
          --gpu-memory-utilization ${GPU_MEM} \
          --max-model-len ${MAX_LEN} \
          --max-num-seqs ${MAX_NUM_SEQS} \
          --enable-prefix-caching \
          --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' \
          >> ${LOG} 2>&1 < /dev/null &"
    local PID=$!
    echo "${PID}"
}

sample_metrics() {
    local OUT=$1
    curl -sf "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
        | grep -E "vllm:gpu_prefix_cache|vllm:prefix_cache" \
        > "${OUT}" || true
}

# Workload-specific spec hook (matches sonnet/chat/code/balanced/sonnet-heavy/
# code-heavy in benchmark_workloads.py — all use target_in=1024 max_tokens=2048).
run_one_bw() {
    local WL=$1 CFG_TAG=$2 SWEEP=$3
    local OUT="${OUTDIR}/${CFG_TAG}_${WL}_s${SWEEP}_bench.json"
    local LOG="${OUTDIR}/${CFG_TAG}_${WL}_s${SWEEP}_bench.log"
    local M0="${OUTDIR}/${CFG_TAG}_${WL}_s${SWEEP}_metrics_before.txt"
    local M1="${OUTDIR}/${CFG_TAG}_${WL}_s${SWEEP}_metrics_after.txt"
    if [[ -s "${OUT}" ]]; then
        echo "[$(ts)] skip existing ${CFG_TAG}_${WL}_s${SWEEP}"
        return 0
    fi
    echo "[$(ts)] --- bench ${CFG_TAG}_${WL}_s${SWEEP} ---"
    sample_metrics "${M0}"
    ${PY} ${BW_PY} \
        --scenario vanilla \
        --workload ${WL} \
        --num-prompts ${NPROMPTS} \
        --target-input-len ${INPUT_LEN} \
        --max-tokens ${OUTPUT_LEN} \
        --concurrency ${CONC} \
        --model "${MODEL}" \
        --sonnet "${DATA}" \
        --seed "$((SWEEP * 1000))" \
        --out "${OUT}" 2>&1 | tee "${LOG}"
    sample_metrics "${M1}"
}

run_config() {
    local CFG_TAG=$1 ENV_PRE=$2
    local STAMP=$(ts_short)
    local BOOT_LOG="${LOGDIR}/step_C_${CFG_TAG}_boot_${STAMP}.log"

    echo "[$(ts)] === STEP C: boot ${CFG_TAG} ==="
    echo "[$(ts)] env: ${ENV_PRE}" | tee -a "${BOOT_LOG}"
    SERVE_PID=$(boot_vllm_port "${CFG_TAG}" "${ENV_PRE}" "${BOOT_LOG}")
    SERVE_PGID=$(ps -o pgid= -p ${SERVE_PID} 2>/dev/null | tr -d ' ')
    echo "${SERVE_PID}"  > "${LOGDIR}/step_C_${CFG_TAG}.pid"
    echo "${SERVE_PGID}" > "${LOGDIR}/step_C_${CFG_TAG}.pgid"

    if ! wait_health ${PORT} 360; then
        echo "[$(ts)] FAIL: vllm not ready for ${CFG_TAG}"
        [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null || true
        cleanup_orphans
        return 1
    fi
    echo "[$(ts)] ${CFG_TAG} vllm ready"

    ${PY} ${BW_PY} \
        --scenario vanilla --workload code \
        --num-prompts 32 --target-input-len 1024 --max-tokens 256 \
        --concurrency 16 --model "${MODEL}" --sonnet "${DATA}" --seed 99999 \
        --out "${LOGDIR}/step_C_${CFG_TAG}_warmup.json" \
        > "${LOGDIR}/step_C_${CFG_TAG}_warmup.log" 2>&1 || true

    trap '
        echo "[$(ts)] trap: kill pgid=${SERVE_PGID}"
        [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null
        cleanup_orphans
        exit 1
    ' INT TERM

    for WL in ${WORKLOADS}; do
        for SWEEP in ${SWEEPS}; do
            run_one_bw "${WL}" "${CFG_TAG}" "${SWEEP}" || true
        done
    done

    echo "[$(ts)] === STEP C: tearing down ${CFG_TAG} ==="
    [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null || true
    [[ -n "${SERVE_PID}" ]]  && kill -9 ${SERVE_PID}    2>/dev/null || true
    sleep 5
    cleanup_orphans
    trap - INT TERM
}

echo "[$(ts)] >>> Step C optionC_v2 start"
for entry in "${CONFIGS[@]}"; do
    CFG_TAG="${entry%%|*}"
    ENV_PRE="${entry#*|}"
    if [[ -n "${ONLY_TAG:-}" && "${CFG_TAG}" != "${ONLY_TAG}" ]]; then
        echo "[$(ts)] skip ${CFG_TAG} (ONLY_TAG=${ONLY_TAG})"
        continue
    fi
    run_config "${CFG_TAG}" "${ENV_PRE}" || echo "[$(ts)] WARN: ${CFG_TAG} returned nonzero"
done
echo "[$(ts)] <<< Step C optionC_v2 complete"
