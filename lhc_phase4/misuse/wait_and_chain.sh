#!/usr/bin/env bash
# Wait for the ap1 sweep loop process to exit, then run chain_all.sh.
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/misuse
ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }
echo "[$(ts)] waiting for ap1 sweep to finish..."
while pgrep -f "WORKLOADS=chat APS=ap1" >/dev/null 2>&1 \
        || pgrep -f "run_misuse_sweep.sh" >/dev/null 2>&1 ; do
    sleep 10
done
echo "[$(ts)] ap1 done — launching chain_all"
bash ${BASE}/chain_all.sh > ${BASE}/chain_all.log 2>&1
echo "[$(ts)] chain_all done"
