#!/usr/bin/env bash
# R2-1 — KV cache dtype NVFP4 (sm_100 native FP4 E2M1). FlashInfer backend.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC2 R2-1 kv-nvfp4 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(--kv-cache-dtype nvfp4)
# Note: vllm/v1/attention/backends/flashinfer.py:625 raises NotImplementedError
# for nvfp4 KV cache. This sweep will boot-fail; kept for documentation.
do_case_nsweep "r2_1_kv_nvfp4" 1

echo "==== HWC2 R2-1 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
