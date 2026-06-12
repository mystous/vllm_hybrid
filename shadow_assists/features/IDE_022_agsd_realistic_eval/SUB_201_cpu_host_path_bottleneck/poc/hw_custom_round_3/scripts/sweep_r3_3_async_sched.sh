#!/usr/bin/env bash
# R3-3 — KV fp8 + async-scheduling enabled.
# Previously CPU overhead masked benefit — re-measure with fp8 baseline.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-3 fp8 + async-sched 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --async-scheduling
)
do_case_nsweep "r3_3_fp8_async" 5
wait_gpu_free || true
