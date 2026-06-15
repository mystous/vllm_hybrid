#!/usr/bin/env bash
# Chain Step B (code variant) → Step C (PREFIX_HOT adaptive).
# Runs sequentially; safe to launch in background once Step A finishes.

set -uo pipefail
HERE="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/scripts"
LOGDIR="/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase/precision_logs"
mkdir -p "${LOGDIR}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

echo "[$(ts)] >>> CHAIN Step B"
bash "${HERE}/step_B_code_variant.sh" > "${LOGDIR}/step_B_runner.log" 2>&1
echo "[$(ts)] === Step B finished, starting Step C ==="

# Force a small pause + GPU cleanup before C
sleep 5
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 5

echo "[$(ts)] >>> CHAIN Step C"
bash "${HERE}/step_C_optionC_v2.sh" > "${LOGDIR}/step_C_runner.log" 2>&1
echo "[$(ts)] <<< CHAIN complete (B+C)"
