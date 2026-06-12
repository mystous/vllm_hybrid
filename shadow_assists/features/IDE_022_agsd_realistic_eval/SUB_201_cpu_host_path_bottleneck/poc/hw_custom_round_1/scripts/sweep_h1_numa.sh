#!/usr/bin/env bash
# H1 — NUMA-aware TP worker pinning (auto-detect, cpunodebind).
# Topology: GPU0-3 -> NUMA0 (cpu 0-55,112-167), GPU4-7 -> NUMA1 (cpu 56-111,168-223).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H1 numa-bind 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --numa-bind
    --numa-bind-nodes 0 0 0 0 1 1 1 1
)
do_case_nsweep "h1_numa_bind" 1

echo "==== HWC1 H1 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
