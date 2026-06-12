#!/usr/bin/env bash
# R2-6 — KV fp8 (R1 winner) + Q (query) quantize for prefill.
# Q quant for prefill phase reduces compute + BW with minimal accuracy loss.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC2 R2-6 fp8 + Q-quant 5-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --kv-cache-dtype fp8
    --attention-config '{"use_prefill_query_quantization":true}'
)
do_case_nsweep "r2_6_fp8_qquant" 5

echo "==== HWC2 R2-6 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
