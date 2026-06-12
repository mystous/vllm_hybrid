#!/usr/bin/env bash
# R3-1 — KV fp8 + FULL_DECODE_ONLY cudagraph (R6C variant w/ 5-sweep).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-1 fp8 + FULL_DECODE_ONLY 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
)
do_case_nsweep "r3_1_fp8_full_decode" 5
wait_gpu_free || true
