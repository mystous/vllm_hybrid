#!/usr/bin/env bash
# R3-2 — KV fp8 + larger max-num-batched-tokens (default ~2048-8192).
# With B200 HBM, can push batch to fit more tokens.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-2 fp8 + batched_tokens=16384 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --max-num-batched-tokens 16384
)
do_case_nsweep "r3_2_fp8_batched16k" 5
wait_gpu_free || true
