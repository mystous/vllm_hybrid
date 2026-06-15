#!/usr/bin/env bash
# Wait for attempt_1 quick sanity to finish, then start attempt_2 sweep
# (only longA + longB with 2 sweeps to keep budget reasonable).
set -uo pipefail

# Wait for attempt_1 lhc_s1 to finish (bench json exists)
for i in $(seq 1 240); do
    if [[ -s /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_1/runs/mem0.30_lhc_s1_bench.json ]]; then
        echo "attempt_1 done at $(date)"
        break
    fi
    sleep 15
done

# Start attempt_2 with reduced scope.
SWEEPS=2 WORKLOADS="longA longB" \
    bash /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_2/run_longctx_sweep.sh \
    > /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_2/run.log 2>&1
echo "attempt_2 done at $(date)"
