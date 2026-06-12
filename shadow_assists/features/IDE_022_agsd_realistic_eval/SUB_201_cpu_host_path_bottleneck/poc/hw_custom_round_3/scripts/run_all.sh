#!/usr/bin/env bash
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"
trap 'echo "[trap]"; exit 130' INT TERM

run_one() {
    echo ""
    echo "===================================================================="
    echo "== $1 @ $(date -u +%FT%TZ)"
    echo "===================================================================="
    bash "$SCRIPT_DIR/$1"
    sleep 5
    wait_gpu_free || true
}

run_one sweep_r3_1_full_decode.sh
run_one sweep_r3_2_max_batched_tokens.sh
run_one sweep_r3_3_async_sched.sh
run_one sweep_r3_4_combo.sh
# R3 stack-on-winner candidates (sp + extra)
run_one sweep_r3_a_sp_async.sh
run_one sweep_r3_b_sp_batched16k.sh
run_one sweep_r3_c_sp_full_decode.sh
run_one sweep_r3_d_sp_maxseqs.sh

echo ""
echo "==== ALL HWC3 levers complete at $(date -u +%FT%TZ) ===="
