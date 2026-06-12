#!/usr/bin/env bash
# H9 — KV-cache dtype NVFP4 (sm_100 native). Roughly 4 bits/elt vs fp8 (8), bf16 (16).
# Activates FlashInfer's NVFP4 KV cache path. Requires sm_100 + flashinfer build with FP4.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H9 kv-nvfp4 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(--kv-cache-dtype nvfp4)
do_case_nsweep "h9_kv_nvfp4" 1

echo "==== HWC1 H9 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
