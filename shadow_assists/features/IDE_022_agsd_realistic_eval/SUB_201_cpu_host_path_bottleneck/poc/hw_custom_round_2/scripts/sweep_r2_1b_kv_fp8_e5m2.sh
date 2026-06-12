#!/usr/bin/env bash
# R2-1b — KV cache fp8_e5m2 (wider exponent range vs fp8_e4m3).
# Round 6 R6D measurement — re-measure with 5 sweeps.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC2 R2-1b kv-fp8_e5m2 5-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(--kv-cache-dtype fp8_e5m2)
do_case_nsweep "r2_1b_kv_fp8_e5m2" 5

echo "==== HWC2 R2-1b complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
