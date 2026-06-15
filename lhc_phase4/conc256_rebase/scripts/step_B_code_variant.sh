#!/usr/bin/env bash
# Step B — Multi-workload generalization for code variant.
#
# For each of {python, rust, json} variant (WORKLOAD_CODE_VARIANT env), run
#  - vanilla : VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
#  - path1   : add VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=...
#  - stack   : path1 + VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1
#              VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_ADAPTIVE=1
# 3 sweeps per (variant × config) over the code workload.
#
# Also samples vllm /metrics endpoint after each bench to capture the
# prefix-cache hit rate (gpu cache hit + queries) so the B-3 cutoff
# (hit_rate >= 60%) can be computed offline.
#
# Outputs:
#   code_variant_runs/<variant>_<config>_code_s<n>_bench.json
#   code_variant_runs/<variant>_<config>_code_s<n>_metrics.txt

set -uo pipefail
HERE="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/scripts"
source "${HERE}/lib_common.sh"

export PORT=8001
export NPROMPTS=500
export INPUT_LEN=1024
export OUTPUT_LEN=2048
export CONC=256

OUTDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/code_variant_runs"
LOGDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/code_variant_logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

BW_PY="/workspace/host_vllm_hybrid/vllm_config_perf/gating/benchmark_workloads.py"
PY="/workspace/vllm_dev_prj/bin/python3"
LIB="/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"

VARIANTS=${VARIANTS:-"python rust json"}
SWEEPS=${SWEEPS:-"1 2 3"}

CONFIGS=(
    "vanilla|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1"
    "path1|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=${LIB}"
    "stack|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=${LIB} VLLM_LHC_REGIME_ADAPTIVE=1"
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

run_one_bw() {
    local VARIANT=$1 CFG_TAG=$2 SWEEP=$3
    local OUT="${OUTDIR}/${VARIANT}_${CFG_TAG}_code_s${SWEEP}_bench.json"
    local LOG="${OUTDIR}/${VARIANT}_${CFG_TAG}_code_s${SWEEP}_bench.log"
    local M0="${OUTDIR}/${VARIANT}_${CFG_TAG}_code_s${SWEEP}_metrics_before.txt"
    local M1="${OUTDIR}/${VARIANT}_${CFG_TAG}_code_s${SWEEP}_metrics_after.txt"
    if [[ -s "${OUT}" ]]; then
        echo "[$(ts)] skip existing ${VARIANT}_${CFG_TAG}_code_s${SWEEP}"
        return 0
    fi
    echo "[$(ts)] --- bench ${VARIANT}_${CFG_TAG}_code_s${SWEEP} ---"
    sample_metrics "${M0}"
    WORKLOAD_CODE_VARIANT=${VARIANT} \
    ${PY} ${BW_PY} \
        --scenario vanilla \
        --workload code \
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
    local BOOT_LOG="${LOGDIR}/step_B_${CFG_TAG}_boot_${STAMP}.log"

    echo "[$(ts)] === STEP B: boot ${CFG_TAG} ==="
    echo "[$(ts)] env: ${ENV_PRE}" | tee -a "${BOOT_LOG}"
    SERVE_PID=$(boot_vllm_port "${CFG_TAG}" "${ENV_PRE}" "${BOOT_LOG}")
    SERVE_PGID=$(ps -o pgid= -p ${SERVE_PID} 2>/dev/null | tr -d ' ')
    echo "${SERVE_PID}"  > "${LOGDIR}/step_B_${CFG_TAG}.pid"
    echo "${SERVE_PGID}" > "${LOGDIR}/step_B_${CFG_TAG}.pgid"
    echo "[$(ts)] booted pid=${SERVE_PID} pgid=${SERVE_PGID}"

    if ! wait_health ${PORT} 360; then
        echo "[$(ts)] FAIL: vllm not ready for ${CFG_TAG}"
        [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null || true
        cleanup_orphans
        return 1
    fi
    echo "[$(ts)] ${CFG_TAG} vllm ready"

    # warmup with the current variant (use python — boot for all variants)
    WORKLOAD_CODE_VARIANT=python \
    ${PY} ${BW_PY} \
        --scenario vanilla --workload code \
        --num-prompts 32 --target-input-len 1024 --max-tokens 256 \
        --concurrency 16 --model "${MODEL}" --sonnet "${DATA}" --seed 99999 \
        --out "${LOGDIR}/step_B_${CFG_TAG}_warmup.json" \
        > "${LOGDIR}/step_B_${CFG_TAG}_warmup.log" 2>&1 || true

    trap '
        echo "[$(ts)] trap: kill pgid=${SERVE_PGID}"
        [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null
        cleanup_orphans
        exit 1
    ' INT TERM

    for VARIANT in ${VARIANTS}; do
        for SWEEP in ${SWEEPS}; do
            run_one_bw "${VARIANT}" "${CFG_TAG}" "${SWEEP}" || true
        done
    done

    echo "[$(ts)] === STEP B: tearing down ${CFG_TAG} ==="
    [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null || true
    [[ -n "${SERVE_PID}" ]]  && kill -9 ${SERVE_PID}    2>/dev/null || true
    sleep 5
    cleanup_orphans
    trap - INT TERM
}

echo "[$(ts)] >>> Step B code-variant start (variants=${VARIANTS}, sweeps=${SWEEPS})"
for entry in "${CONFIGS[@]}"; do
    CFG_TAG="${entry%%|*}"
    ENV_PRE="${entry#*|}"
    if [[ -n "${ONLY_TAG:-}" && "${CFG_TAG}" != "${ONLY_TAG}" ]]; then
        echo "[$(ts)] skip ${CFG_TAG} (ONLY_TAG=${ONLY_TAG})"
        continue
    fi
    run_config "${CFG_TAG}" "${ENV_PRE}" || echo "[$(ts)] WARN: ${CFG_TAG} returned nonzero"
done
echo "[$(ts)] <<< Step B code-variant complete"
