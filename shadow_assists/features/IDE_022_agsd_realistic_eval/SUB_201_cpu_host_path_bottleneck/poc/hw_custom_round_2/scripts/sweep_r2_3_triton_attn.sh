#!/usr/bin/env bash
# R2-3 — Force TRITON_ATTN backend (vs default FLASHINFER).
# Previous round_8 B3_8gpu sweep showed TRITON_ATTN gave -71% on 8 GPU FULL (regression).
# But baseline now has B3_FaP — re-measure on current baseline to confirm.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC2 R2-3 triton-attn 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=("VLLM_ATTENTION_BACKEND=TRITON_ATTN")
declare -a EXTRA_CLI=(--kv-cache-dtype fp8)
do_case_nsweep "r2_3_fp8_triton_attn" 1

echo "==== HWC2 R2-3 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
