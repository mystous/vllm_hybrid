#!/usr/bin/env bash
# H4 — PyTorch CUDA allocator: expandable_segments=True.
# Should reduce fragmentation on B200 HBM3e and allow large KV reservations
# without retry-on-fragment fallback. May reduce alloc-time stalls.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H4 expand-seg 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
declare -a EXTRA_CLI=()
do_case_nsweep "h4_expand_seg" 1

echo "==== HWC1 H4 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
