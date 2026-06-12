#!/usr/bin/env bash
# Round 2 — KV fp8 (R1 winner) + 5 candidate combinations.
# R2-2 is just fp8 (identical to R1 H8) → skip.
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

# R2-6 (fp8 + Q quant) — most promising (additional BW savings on Q tensor)
run_one sweep_r2_6_kv_fp8_qquant.sh
# R2-4 (fp8 + fuse_norm_quant) — kernel fusion
run_one sweep_r2_4_fuse_norm.sh
# R2-5 (fp8 + enable_sp) — seq parallel for TP comm reduction
run_one sweep_r2_5_enable_sp.sh
# R2-3 (fp8 + TRITON_ATTN) — attention backend swap
run_one sweep_r2_3_triton_attn.sh
# R2-1b (fp8_e5m2 5-sweep) — alt fp8 variant
run_one sweep_r2_1b_kv_fp8_e5m2.sh

echo ""
echo "==== ALL HWC2 levers complete at $(date -u +%FT%TZ) ===="
