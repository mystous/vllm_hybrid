#!/usr/bin/env bash
# Step 5 — LHC integrated stack (Option A + Path 1 + Option C all on).
# 6 workloads × 3 sweeps.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/lib_common.sh"

OUTDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/stack_runs"
LOGDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

SWEEPS=${SWEEPS:-3}
WORKLOADS=${WORKLOADS:-${WORKLOADS_DEFAULT}}

BOOT_LOG="${LOGDIR}/step5_stack_boot_$(ts_short).log"
LIB="/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"
ENV_PRE="VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=${LIB} VLLM_LHC_REGIME_ADAPTIVE=1"

echo "[$(ts)] === Step 5 boot stack (A + Path1 + C, conc256) ===" | tee "${BOOT_LOG}"
echo "[$(ts)] env: ${ENV_PRE}" | tee -a "${BOOT_LOG}"
SERVE_PID=$(boot_vllm "step5_stack" "${ENV_PRE}" "${BOOT_LOG}")
echo "${SERVE_PID}" > "${LOGDIR}/step5_stack.pid"

if ! wait_health ${PORT} 240; then
    echo "[$(ts)] vllm not ready" | tee -a "${BOOT_LOG}"
    [[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null || true
    cleanup_orphans
    exit 1
fi
echo "[$(ts)] vllm ready" | tee -a "${BOOT_LOG}"

${VLLM_BIN} bench serve \
    --model "${MODEL}" --dataset-name sonnet --dataset-path "${DATA}" \
    --sonnet-input-len 512 --sonnet-output-len 64 \
    --num-prompts 32 --max-concurrency 16 --port ${PORT} \
    > "${LOGDIR}/step5_warmup.log" 2>&1 || true

trap 'echo "[$(ts)] trap: kill pid=${SERVE_PID}"; [[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null; cleanup_orphans; exit 1' INT TERM

for WORKLOAD in ${WORKLOADS}; do
    for SWEEP in $(seq 1 ${SWEEPS}); do
        TAG="stack_${WORKLOAD}_s${SWEEP}"
        if [[ -s "${OUTDIR}/${TAG}.json" ]]; then
            echo "[$(ts)] skip existing ${TAG}"
            continue
        fi
        echo "[$(ts)] --- bench ${TAG} ---" | tee -a "${BOOT_LOG}"
        run_bench "${TAG}" "${WORKLOAD}" "${OUTDIR}" "$((SWEEP * 1000))"
    done
done

echo "[$(ts)] === Step 5 done, killing pid=${SERVE_PID} ===" | tee -a "${BOOT_LOG}"
[[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null || true
sleep 3
cleanup_orphans
echo "[$(ts)] Step 5 complete."
