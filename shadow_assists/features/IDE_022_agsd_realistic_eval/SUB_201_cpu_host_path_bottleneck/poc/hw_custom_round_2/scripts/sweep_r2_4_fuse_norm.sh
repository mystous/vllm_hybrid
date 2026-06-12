#!/usr/bin/env bash
# R2-4 — KV fp8 (Round 1 winner) + inductor pass: fuse_norm_quant + fuse_act_quant + fuse_attn_quant.
# Stack on top of fp8 to push past +10%.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC2 R2-4 fp8 + fuse-norm-quant 5-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --kv-cache-dtype fp8
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_attn_quant":true}}'
)
do_case_nsweep "r2_4_fp8_fuse_norm" 5

echo "==== HWC2 R2-4 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
