#!/usr/bin/env bash
# H10 — torch.compile: assume_32_bit_indexing=True for cudagraph kernels.
# B200's max KV (~5M tokens × layer × head_dim) still fits in int32 — eliminates
# int64 mul/div in Triton index computation, which can be ~20% faster on Hopper/Blackwell.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H10 int32-idx 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

# Override compilation-config to add dynamic_shapes_config.assume_32_bit_indexing
declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","dynamic_shapes_config":{"assume_32_bit_indexing":true}}'
)
# Note: lib_common already adds --compilation-config first; need a patched approach.
echo "[H10] Will override via overriding LAST flag (vllm CLI keeps last value)"
do_case_nsweep "h10_int32_idx" 1

echo "==== HWC1 H10 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
