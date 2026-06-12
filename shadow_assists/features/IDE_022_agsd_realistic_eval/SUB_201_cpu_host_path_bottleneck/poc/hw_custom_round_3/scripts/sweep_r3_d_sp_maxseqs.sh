#!/usr/bin/env bash
# R3-D — fp8 + enable_sp + max-num-seqs=256 (4x default 64).
# Increase concurrent sequences to push KV cache utilization on B200.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC3 R3-D fp8 + sp + maxseqs=256 5-sweep ===="
wait_gpu_free || true
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"enable_sp":true,"fuse_gemm_comms":true}}'
    --max-num-seqs 256
)
do_case_nsweep "r3_d_fp8_sp_maxseqs256" 5
wait_gpu_free || true
