#!/usr/bin/env bash
# After attempt_2 longA (2 sweeps × {vanilla, lhc}) completes,
# stop the attempt_2 sweep early and start attempt_3.
set -uo pipefail

# Wait until attempt_2 lhc longA s2 bench json exists (or 30 min timeout).
for i in $(seq 1 240); do
    if [[ -s /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_2/runs/longA_lhc_s2_bench.json ]]; then
        echo "attempt_2 longA all done at $(date)"
        break
    fi
    sleep 15
done

# Kill the attempt_2 sweep script & its vllm.
pkill -9 -f "run_longctx_sweep.sh" 2>/dev/null || true
sleep 3
SERVE_PIDS=$(pgrep -f "vllm serve.*--port 8500" 2>/dev/null || true)
for p in $SERVE_PIDS; do kill -9 "$p" 2>/dev/null; done
sleep 8

# Start attempt_3 (ext1 + ext2, 2 sweeps).
SWEEPS=2 WORKLOADS="ext1 ext2" \
    bash /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_3/run_force_swap_sweep.sh \
    > /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_3/run.log 2>&1
echo "attempt_3 done at $(date)"
