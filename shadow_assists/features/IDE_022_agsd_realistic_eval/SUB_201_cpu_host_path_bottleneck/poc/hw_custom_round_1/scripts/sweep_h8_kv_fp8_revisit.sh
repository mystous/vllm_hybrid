#!/usr/bin/env bash
# H8 — KV cache fp8 revisit with 5-sweep statistical power.
# Round 6 single-sweep showed +3.38% — re-measure with 5 sweeps + accuracy gate later.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H8 kv-fp8 5-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(--kv-cache-dtype fp8)
do_case_nsweep "h8_kv_fp8" 5

echo "==== HWC1 H8 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
