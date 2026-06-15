#!/usr/bin/env bash
# Step 1 — conc=256 vanilla baseline.
# 6 workloads × 5 sweeps under a single vllm boot.
#
# env:
#   VLLM_PREFETCH_TOKENIZE=1
#   VLLM_BURST_AWARE_ADMISSION=1
# (L2/L10 from "Optimal Config" — no LHC flags).

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/lib_common.sh"

OUTDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/vanilla_runs"
LOGDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

SWEEPS=${SWEEPS:-5}
WORKLOADS=${WORKLOADS:-${WORKLOADS_DEFAULT}}

BOOT_LOG="${LOGDIR}/step1_vanilla_boot_$(ts_short).log"
ENV_PRE="VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1"

echo "[$(ts)] === Step 1 boot vanilla (conc256) ===" | tee "${BOOT_LOG}"
echo "[$(ts)] env: ${ENV_PRE}" | tee -a "${BOOT_LOG}"
SERVE_PID=$(boot_vllm "step1_vanilla" "${ENV_PRE}" "${BOOT_LOG}")
echo "${SERVE_PID}" > "${LOGDIR}/step1_vanilla.pid"
SERVE_PGID=$(ps -o pgid= -p ${SERVE_PID} 2>/dev/null | tr -d ' ')
echo "${SERVE_PGID}" > "${LOGDIR}/step1_vanilla.pgid"

echo "[$(ts)] boot pid=${SERVE_PID} pgid=${SERVE_PGID}, waiting for health..." | tee -a "${BOOT_LOG}"
if ! wait_health ${PORT} 240; then
    echo "[$(ts)] vllm not ready" | tee -a "${BOOT_LOG}"
    [[ -n "${SERVE_PGID}" ]] && kill -9 -${SERVE_PGID} 2>/dev/null || true
    cleanup_orphans
    exit 1
fi
echo "[$(ts)] vllm ready" | tee -a "${BOOT_LOG}"

# warmup — 1 small bench to stabilize CUDA graphs etc.
WARM="step1_warmup"
echo "[$(ts)] warmup..." | tee -a "${BOOT_LOG}"
${VLLM_BIN} bench serve \
    --model "${MODEL}" \
    --dataset-name sonnet \
    --dataset-path "${DATA}" \
    --sonnet-input-len 512 --sonnet-output-len 64 \
    --num-prompts 32 --max-concurrency 16 --port ${PORT} \
    > "${LOGDIR}/step1_warmup.log" 2>&1 || true

trap 'echo "[$(ts)] trap: kill pid=${SERVE_PID}"; [[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null; cleanup_orphans; exit 1' INT TERM

for WORKLOAD in ${WORKLOADS}; do
    for SWEEP in $(seq 1 ${SWEEPS}); do
        TAG="vanilla_${WORKLOAD}_s${SWEEP}"
        if [[ -s "${OUTDIR}/${TAG}.json" ]]; then
            echo "[$(ts)] skip existing ${TAG}"
            continue
        fi
        echo "[$(ts)] --- bench ${TAG} ---" | tee -a "${BOOT_LOG}"
        run_bench "${TAG}" "${WORKLOAD}" "${OUTDIR}" "$((SWEEP * 1000))"
    done
done

echo "[$(ts)] === Step 1 done, killing pid=${SERVE_PID} ===" | tee -a "${BOOT_LOG}"
[[ -n "${SERVE_PID}" ]] && kill -9 ${SERVE_PID} 2>/dev/null || true
sleep 3
cleanup_orphans
echo "[$(ts)] Step 1 complete."
