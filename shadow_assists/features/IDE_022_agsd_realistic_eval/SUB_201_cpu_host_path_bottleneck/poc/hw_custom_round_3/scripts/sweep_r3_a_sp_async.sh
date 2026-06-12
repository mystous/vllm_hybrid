#!/usr/bin/env bash
# R3-A — fp8 + enable_sp (R2 winner) + async-scheduling.
# Build on R2-5 (+4.25%) by adding host-side overlap with async-sched.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-A fp8 + sp + async 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"enable_sp":true,"fuse_gemm_comms":true}}'
    --async-scheduling
)
do_case_nsweep "r3_a_fp8_sp_async" 5
wait_gpu_free || true
