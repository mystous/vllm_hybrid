#!/usr/bin/env bash
# Run ap1->ap5 sweeps sequentially. Each ap uses chat workload, 2 sweeps,
# 4 cells × ~3min = ~12min per ap.
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/misuse
ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }
for AP in ap2 ap3 ap4 ap5; do
    echo "[$(ts)] === starting chain ${AP} ==="
    WORKLOADS=chat APS=${AP} SWEEPS=3 bash ${BASE}/run_misuse_sweep.sh \
        > ${BASE}/sweep_${AP}.log 2>&1
done
# After all aps done, add a 3rd sweep for ap1 to match.
echo "[$(ts)] === starting ap1 sweep 3 catchup ==="
WORKLOADS=chat APS=ap1 SWEEPS=3 bash ${BASE}/run_misuse_sweep.sh \
    > ${BASE}/sweep_ap1_s3.log 2>&1
echo "[$(ts)] all chain sweeps done."
