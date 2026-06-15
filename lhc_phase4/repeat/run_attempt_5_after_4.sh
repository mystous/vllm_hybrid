#!/usr/bin/env bash
# After attempt_4 eag1 lhc s2 done, start attempt_5 (kvcap1 with code fix).
set -uo pipefail

for i in $(seq 1 240); do
    if [[ -s /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_4/runs/eag1_lhc_s2_bench.json ]]; then
        echo "attempt_4 done at $(date)"
        break
    fi
    sleep 15
done

pkill -9 -f "run_eager_swap.sh" 2>/dev/null || true
sleep 3
SERVE_PIDS=$(pgrep -f "vllm serve.*--port 8500" 2>/dev/null || true)
for p in $SERVE_PIDS; do kill -9 "$p" 2>/dev/null; done
sleep 8

SWEEPS=2 WORKLOADS="kvcap1" \
    bash /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_5/run_kvcap_fix_swap.sh \
    > /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_5/run.log 2>&1
echo "attempt_5 done at $(date)"
