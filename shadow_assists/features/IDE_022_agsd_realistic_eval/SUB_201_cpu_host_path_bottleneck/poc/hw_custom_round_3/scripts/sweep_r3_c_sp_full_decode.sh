#!/usr/bin/env bash
# R3-C — fp8 + enable_sp + FULL_DECODE_ONLY cudagraph.
# Hypothesis: FULL_DECODE_ONLY may reduce graph capture overhead vs FULL_AND_PIECEWISE.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-C fp8 + sp + FULL_DECODE_ONLY 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","pass_config":{"enable_sp":true,"fuse_gemm_comms":true}}'
)
do_case_nsweep "r3_c_fp8_sp_fulldecode" 5
wait_gpu_free || true
