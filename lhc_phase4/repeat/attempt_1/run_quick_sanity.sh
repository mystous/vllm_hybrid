#!/usr/bin/env bash
# Quick sanity: mem=0.30 vanilla + lhc, 1 sweep each. Verify KV pressure +
# NEO swap fires. If fires, proceed to full sweep. If no swap, drop mem
# further.
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_1
SWEEPS=1 MEMS="0.30" CONFIGS="vanilla lhc" "${BASE}/run_mem_util_sweep.sh"
