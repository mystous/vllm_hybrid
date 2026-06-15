#!/usr/bin/env bash
# Step A — Precision sweep for code workload.
#
# Adds s4 / s5 to path1/optionC/stack on code workload only, to extend the
# existing 3-sweep batch up to 5-sweep and tighten the 95% CI on the +10.67%
# path1 finding.
#
# Outputs:
#   precision_runs/{path1,optionC,stack}_code_s{4,5}_bench.json
# Boot logs:
#   precision_logs/step_A_<config>_boot_<ts>.log
#
# env baseline (matches step3/4/5):
#   VLLM_PREFETCH_TOKENIZE=1
#   VLLM_BURST_AWARE_ADMISSION=1
# + path1: VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=...
# + optionC: VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_ADAPTIVE=1
# + stack: path1 + optionC envs combined
#
# Each config gets its own vllm-serve boot (port 8001), warmup, 2 benches
# (seeds 4000, 5000), and graceful teardown.

set -uo pipefail

HERE="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/scripts"
source "${HERE}/lib_common.sh"

# Override defaults
export PORT=8001
export NPROMPTS=500
export INPUT_LEN=1024
export OUTPUT_LEN=2048
export CONC=256

OUTDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/precision_runs"
LOGDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/precision_logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

BW_PY="/workspace/host_vllm_hybrid/vllm_config_perf/gating/benchmark_workloads.py"
PY="/workspace/vllm_dev_prj/bin/python3"
LIB="/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"

if [[ ! -f "${LIB}" ]]; then
    echo "[$(ts)] FATAL: missing AMX C3 lib at ${LIB}"; exit 1
fi
if [[ ! -f "${BW_PY}" ]]; then
    echo "[$(ts)] FATAL: missing benchmark_workloads.py at ${BW_PY}"; exit 1
fi

# Each config: (TAG, ENV_PRE)
CONFIGS=(
    "path1|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=${LIB}"
    "optionC|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_ADAPTIVE=1"
    "stack|VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=${LIB} VLLM_LHC_REGIME_ADAPTIVE=1"
)

SWEEPS=${SWEEPS:-"4 5"}

# Boot vllm on PORT with given env in a NEW process group via setsid.
# This is critical: kill -9 -<pgid> on teardown must NOT take down the runner.
boot_vllm_port() {
    local TAG=$1 ENV_PRE=$2 LOG=$3
    cleanup_orphans
    # `setsid` makes the child the leader of a new session+PG, isolating it
    # from the runner's PG so teardown can safely kill -<pgid>.
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

run_one_bw() {
    local CFG_TAG=$1 SWEEP=$2
    local OUT="${OUTDIR}/${CFG_TAG}_code_s${SWEEP}_bench.json"
    local LOG="${OUTDIR}/${CFG_TAG}_code_s${SWEEP}_bench.log"
    if [[ -s "${OUT}" ]]; then
        echo "[$(ts)] skip existing ${CFG_TAG}_code_s${SWEEP}"
        return 0
    fi
    echo "[$(ts)] --- bench ${CFG_TAG}_code_s${SWEEP} ---"
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
}

run_config() {
    local CFG_TAG=$1 ENV_PRE=$2
    local STAMP=$(ts_short)
    local BOOT_LOG="${LOGDIR}/step_A_${CFG_TAG}_boot_${STAMP}.log"

    echo "[$(ts)] === STEP A: boot ${CFG_TAG} ==="
    echo "[$(ts)] env: ${ENV_PRE}"     | tee -a "${BOOT_LOG}"
    echo "[$(ts)] === BOOT ${CFG_TAG} ===" >> "${BOOT_LOG}"

    SERVE_PID=$(boot_vllm_port "${CFG_TAG}" "${ENV_PRE}" "${BOOT_LOG}")
    SERVE_PGID=$(ps -o pgid= -p ${SERVE_PID} 2>/dev/null | tr -d ' ')
    echo "${SERVE_PID}" > "${LOGDIR}/step_A_${CFG_TAG}.pid"
    echo "${SERVE_PGID}" > "${LOGDIR}/step_A_${CFG_TAG}.pgid"
    echo "[$(ts)] booted pid=${SERVE_PID} pgid=${SERVE_PGID}, awaiting health..."

    if ! wait_health ${PORT} 360; then
        echo "[$(ts)] FAIL: vllm not ready for ${CFG_TAG}"
        [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null || true
        [[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null || true
        cleanup_orphans
        return 1
    fi
    echo "[$(ts)] ${CFG_TAG} vllm ready"

    # Brief warmup via benchmark_workloads (32 prompts).
    ${PY} ${BW_PY} \
        --scenario vanilla --workload code \
        --num-prompts 32 --target-input-len 1024 --max-tokens 256 \
        --concurrency 16 --model "${MODEL}" --sonnet "${DATA}" --seed 99999 \
        --out "${LOGDIR}/step_A_${CFG_TAG}_warmup.json" \
        > "${LOGDIR}/step_A_${CFG_TAG}_warmup.log" 2>&1 || true

    trap '
        echo "[$(ts)] trap: kill pgid=${SERVE_PGID}"
        [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null
        [[ -n "${SERVE_PID}" ]]  && kill -9 ${SERVE_PID} 2>/dev/null
        cleanup_orphans
        exit 1
    ' INT TERM

    local rc=0
    for SWEEP in ${SWEEPS}; do
        run_one_bw "${CFG_TAG}" "${SWEEP}" || rc=$?
    done

    echo "[$(ts)] === STEP A: tearing down ${CFG_TAG} pid=${SERVE_PID} pgid=${SERVE_PGID} ==="
    [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null || true
    [[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null || true
    sleep 5
    cleanup_orphans
    trap - INT TERM
    return $rc
}

# === main loop ===
echo "[$(ts)] >>> Step A precision (code workload) start"
for entry in "${CONFIGS[@]}"; do
    CFG_TAG="${entry%%|*}"
    ENV_PRE="${entry#*|}"
    if [[ -n "${ONLY_TAG:-}" && "${CFG_TAG}" != "${ONLY_TAG}" ]]; then
        echo "[$(ts)] skip ${CFG_TAG} (ONLY_TAG=${ONLY_TAG})"
        continue
    fi
    run_config "${CFG_TAG}" "${ENV_PRE}" || echo "[$(ts)] WARN: ${CFG_TAG} returned nonzero"
done
echo "[$(ts)] <<< Step A precision complete"
