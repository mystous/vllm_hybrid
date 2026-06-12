#!/usr/bin/env bash
# HWC1 Round-1 — sweep all 6 candidates sequentially (1 sweep each, ~6 min each).
# Skips a case if its result json already exists.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap] interrupted"; exit 130' INT TERM

run_one() {
    local script=$1
    echo ""
    echo "===================================================================="
    echo "== $script @ $(date -u +%FT%TZ)"
    echo "===================================================================="
    bash "$SCRIPT_DIR/$script"
    sleep 5
    wait_gpu_free || true
}

run_one sweep_h1_numa.sh
run_one sweep_h2_numa_physcpu.sh
run_one sweep_h3_stream_prio.sh
run_one sweep_h4_expand_seg.sh
run_one sweep_h6_jemalloc.sh
run_one sweep_h7_malloc_arena.sh
run_one sweep_h8_kv_fp8_revisit.sh
run_one sweep_h10_int32_idx.sh

echo ""
echo "==== ALL HWC1 levers complete at $(date -u +%FT%TZ) ===="
python3 "$SCRIPT_DIR/analyze.py" || true
