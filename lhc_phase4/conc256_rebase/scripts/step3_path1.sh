#!/usr/bin/env bash
# Step 3 — LHC Path 1 (AMX C3 prefix-cache hash production lib).
# env adds: VLLM_LHC_AMX_C3_PREFIX=1, VLLM_LHC_AMX_C3_LIB=...
# Workloads: chat, sonnet (per task: chat_prefix not in our 6-set so use chat as proxy).
# 3 sweeps.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/lib_common.sh"

OUTDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/path1_runs"
LOGDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

SWEEPS=${SWEEPS:-3}
WORKLOADS=${WORKLOADS:-"chat sonnet"}

BOOT_LOG="${LOGDIR}/step3_path1_boot_$(ts_short).log"
LIB="/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"
if [[ ! -f "${LIB}" ]]; then
    echo "[$(ts)] missing AMX C3 lib at ${LIB}" | tee "${BOOT_LOG}"
    exit 1
fi

ENV_PRE="VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=${LIB}"

echo "[$(ts)] === Step 3 boot Path 1 (AMX C3 prefix, conc256) ===" | tee "${BOOT_LOG}"
echo "[$(ts)] env: ${ENV_PRE}" | tee -a "${BOOT_LOG}"
SERVE_PID=$(boot_vllm "step3_path1" "${ENV_PRE}" "${BOOT_LOG}")
echo "${SERVE_PID}" > "${LOGDIR}/step3_path1.pid"

echo "[$(ts)] boot pid=${SERVE_PID}, waiting for health..." | tee -a "${BOOT_LOG}"
if ! wait_health ${PORT} 240; then
    echo "[$(ts)] vllm not ready" | tee -a "${BOOT_LOG}"
    [[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null || true
    cleanup_orphans
    exit 1
fi
echo "[$(ts)] vllm ready" | tee -a "${BOOT_LOG}"

# warmup
${VLLM_BIN} bench serve \
    --model "${MODEL}" --dataset-name sonnet --dataset-path "${DATA}" \
    --sonnet-input-len 512 --sonnet-output-len 64 \
    --num-prompts 32 --max-concurrency 16 --port ${PORT} \
    > "${LOGDIR}/step3_warmup.log" 2>&1 || true

trap 'echo "[$(ts)] trap: kill pid=${SERVE_PID}"; [[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null; cleanup_orphans; exit 1' INT TERM

for WORKLOAD in ${WORKLOADS}; do
    for SWEEP in $(seq 1 ${SWEEPS}); do
        TAG="path1_${WORKLOAD}_s${SWEEP}"
        if [[ -s "${OUTDIR}/${TAG}.json" ]]; then
            echo "[$(ts)] skip existing ${TAG}"
            continue
        fi
        echo "[$(ts)] --- bench ${TAG} ---" | tee -a "${BOOT_LOG}"
        run_bench "${TAG}" "${WORKLOAD}" "${OUTDIR}" "$((SWEEP * 1000))"
    done
done

echo "[$(ts)] === Step 3 done, killing pid=${SERVE_PID} ===" | tee -a "${BOOT_LOG}"
[[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null || true
sleep 3
cleanup_orphans
echo "[$(ts)] Step 3 complete."
