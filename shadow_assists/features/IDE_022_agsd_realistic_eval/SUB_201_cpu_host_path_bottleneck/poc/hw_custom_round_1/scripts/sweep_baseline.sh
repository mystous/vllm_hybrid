#!/usr/bin/env bash
# hw_custom_round_1 — baseline 5-sweep
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 baseline 5-sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=()
do_case_nsweep "baseline" 5

echo "==== HWC1 baseline complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
