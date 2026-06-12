#!/usr/bin/env bash
# H6 — LD_PRELOAD jemalloc to replace glibc ptmalloc for the API server proc.
# jemalloc reduces lock contention and fragmentation on multi-thread Python.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

JEM=/workspace/vllm_dev_prj/lib/python3.12/site-packages/ray/core/libjemalloc.so

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H6 jemalloc 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

if [ ! -f "$JEM" ]; then
    echo "[H6] jemalloc not found at $JEM — abort"
    exit 1
fi

declare -a EXTRA_ENV=("LD_PRELOAD=$JEM")
declare -a EXTRA_CLI=()
do_case_nsweep "h6_jemalloc" 1

echo "==== HWC1 H6 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
