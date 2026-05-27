#!/usr/bin/env bash
# SUB_185 chain: OFF -> cleanup -> ON
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/SUB_185_cold_kv_long_context
CHAIN_LOG=${BASE}/logs/chain.log
mkdir -p "${BASE}/logs"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

echo "[$(ts)] chain starting" | tee -a "${CHAIN_LOG}"

# Stage 1: OFF
echo "[$(ts)] OFF launching" | tee -a "${CHAIN_LOG}"
bash "${BASE}/launcher.sh" off > "${BASE}/logs/launcher_off_outer.log" 2>&1
RC_OFF=$?
echo "[$(ts)] OFF returned rc=${RC_OFF}" | tee -a "${CHAIN_LOG}"

if ! grep -q "SUB_185 mode=off done" "${BASE}/logs/main_off.log" 2>/dev/null; then
    echo "[$(ts)] OFF did not write done sentinel — aborting before ON" | tee -a "${CHAIN_LOG}"
    exit 1
fi

# Hard cleanup between modes
pgrep -f "vllm serve.*32B|VLLM::|eval/monitor.py|cold_kv_firer" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 15

# Stage 2: ON
echo "[$(ts)] ON launching" | tee -a "${CHAIN_LOG}"
bash "${BASE}/launcher.sh" on > "${BASE}/logs/launcher_on_outer.log" 2>&1
RC_ON=$?
echo "[$(ts)] ON returned rc=${RC_ON}" | tee -a "${CHAIN_LOG}"

echo "[$(ts)] SUB_185 chain done" | tee -a "${CHAIN_LOG}"
