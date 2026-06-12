#!/usr/bin/env bash
# R2-5 — KV fp8 (R1 winner) + enable_sp + fuse_gemm_comms.
# SP splits LN/RMSNorm along seq dim across TP ranks → less AR + memory.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC2 R2-5 fp8 + enable-sp 5-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --kv-cache-dtype fp8
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"enable_sp":true,"fuse_gemm_comms":true}}'
)
do_case_nsweep "r2_5_fp8_sp" 5

echo "==== HWC2 R2-5 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
