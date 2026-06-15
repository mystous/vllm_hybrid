#!/usr/bin/env bash
# After attempt_2 longA lhc s2 done, skip attempt_3 (OOB drop issue),
# go directly to attempt_4 (eager mode to avoid stale kv_cap).
set -uo pipefail

for i in $(seq 1 240); do
    if [[ -s /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_2/runs/longA_lhc_s2_bench.json ]]; then
        echo "attempt_2 longA lhc s2 done at $(date)"
        break
    fi
    sleep 15
done

# Cancel chain23 if still running (it would start attempt_3 ext).
pkill -9 -f "run_attempt_3_after_2_longA.sh" 2>/dev/null || true
pkill -9 -f "run_longctx_sweep.sh" 2>/dev/null || true
pkill -9 -f "run_force_swap_sweep.sh" 2>/dev/null || true
sleep 3
SERVE_PIDS=$(pgrep -f "vllm serve.*--port 8500" 2>/dev/null || true)
for p in $SERVE_PIDS; do kill -9 "$p" 2>/dev/null; done
sleep 8

SWEEPS=2 WORKLOADS="eag1" \
    bash /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_4/run_eager_swap.sh \
    > /workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_4/run.log 2>&1
echo "attempt_4 done at $(date)"
