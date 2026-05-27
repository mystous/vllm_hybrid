#!/usr/bin/env bash
# SUB_194 — orchestrator: chain all 3 levers × OFF/ON × 3 runs.
# Each (lever, mode) cycle = one fresh vllm boot followed by 3 back-to-back agsd-gated runs.
set -uo pipefail
BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_015_cpu_extreme_util/SUB_194_multi_run_variance
LAUNCH="${BASE}/launcher.sh"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

echo "[$(ts)] SUB_194 runner start"
for LEVER in L183 L188 L190; do
    for MODE in off on; do
        echo "[$(ts)] >>>>> cycle ${LEVER} ${MODE}"
        bash "${LAUNCH}" "${LEVER}" "${MODE}" 3
        echo "[$(ts)] <<<<< cycle ${LEVER} ${MODE} done; cooldown 15s"
        sleep 15
    done
done
echo "[$(ts)] SUB_194 runner finished all cycles"
