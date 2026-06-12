#!/usr/bin/env bash
# R3-B — fp8 + enable_sp + max-num-batched-tokens=16384 (4x default 4096).
# Allow much larger batches given B200's 183GB HBM.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-B fp8 + sp + batched16k 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"enable_sp":true,"fuse_gemm_comms":true}}'
    --max-num-batched-tokens 16384
)
do_case_nsweep "r3_b_fp8_sp_batched16k" 5
wait_gpu_free || true
