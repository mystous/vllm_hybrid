#!/usr/bin/env bash
# R2-2 — KV cache fp8 with 5-sweep + paired stats (vs Round 1 baseline 22078±152).
# R6A previously showed +3.38% in single-sweep; verify with stat power.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC2 R2-2 kv-fp8 5-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(--kv-cache-dtype fp8)
do_case_nsweep "r2_2_kv_fp8" 5

echo "==== HWC2 R2-2 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
