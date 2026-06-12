#!/usr/bin/env bash
# H7 — glibc malloc tuning: limit arenas (less contention), eager trim.
# MALLOC_ARENA_MAX=2 vs default (8 × ncpu) — reduces per-arena overhead.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H7 malloc-arena 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=(
    "MALLOC_ARENA_MAX=2"
    "MALLOC_MMAP_THRESHOLD_=131072"
)
declare -a EXTRA_CLI=()
do_case_nsweep "h7_malloc_arena" 1

echo "==== HWC1 H7 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
