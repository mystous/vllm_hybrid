#!/usr/bin/env bash
# R3-4 — Combo: KV fp8 + async-sched + max-num-batched-tokens=8192 + max-num-seqs=256.
# Aggressive throughput-oriented stacking.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-4 fp8 + combo 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --async-scheduling
    --max-num-batched-tokens 8192
    --max-num-seqs 256
)
do_case_nsweep "r3_4_fp8_combo" 5
wait_gpu_free || true
