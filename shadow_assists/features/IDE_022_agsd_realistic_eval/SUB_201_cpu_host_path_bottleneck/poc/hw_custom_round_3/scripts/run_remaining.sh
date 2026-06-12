#!/usr/bin/env bash
# Resume R3 from R3-3 (skip R3-2 which crashed engine with batched=16384).
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

run_one sweep_r3_3_async_sched.sh
run_one sweep_r3_4_combo.sh
run_one sweep_r3_a_sp_async.sh
run_one sweep_r3_c_sp_full_decode.sh
run_one sweep_r3_d_sp_maxseqs.sh
# R3-B (batched 16384 + sp) skipped — same crash risk as R3-2

echo ""
echo "==== R3 remaining levers complete at $(date -u +%FT%TZ) ===="
/workspace/vllm_dev_prj/bin/python "$SCRIPT_DIR/analyze.py"
