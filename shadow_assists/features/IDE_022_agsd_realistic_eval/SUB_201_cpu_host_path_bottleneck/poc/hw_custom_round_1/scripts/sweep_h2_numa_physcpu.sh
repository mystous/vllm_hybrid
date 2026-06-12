#!/usr/bin/env bash
# H2 — NUMA-bind with explicit physcpubind (one 14-core block per GPU).
# 112 physical cores (56C/socket). 14 cores per GPU.
# GPU0: 0-13, GPU1: 14-27, GPU2: 28-41, GPU3: 42-55 (NUMA0)
# GPU4: 56-69, GPU5: 70-83, GPU6: 84-97, GPU7: 98-111 (NUMA1)
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H2 numa+physcpu 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=(
    --numa-bind
    --numa-bind-nodes 0 0 0 0 1 1 1 1
    --numa-bind-cpus "0-13" "14-27" "28-41" "42-55" "56-69" "70-83" "84-97" "98-111"
)
do_case_nsweep "h2_numa_physcpu" 1

echo "==== HWC1 H2 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
