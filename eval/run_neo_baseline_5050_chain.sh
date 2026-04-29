#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# A→B→C chain — 50:50 input/output baseline at 5000 / 1000 / 500 prompts.
# Runs sequentially so they share the GPU exclusively.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

COMMON_ARGS=(
    --model llama-70b
    --tensor-parallel-size 8
    --max-model-len 16384
    --max-num-seqs 256
    --max-tokens 8192
    --target-input-len 8192
)

run_scenario() {
    local n=$1
    local tag=$2
    echo "============================================================"
    echo "STARTING ${tag} — n=${n} input=8192 output=8192"
    echo "============================================================"
    LOG_FILE=/tmp/neo_baseline_${tag}.log \
    OUTPUT_FILE=/tmp/neo_baseline_${tag}.json \
        bash "$SCRIPT_DIR/run_neo_baseline.sh" \
            "${COMMON_ARGS[@]}" \
            --num-prompts "$n"
    echo "============================================================"
    echo "DONE ${tag}: $(cat /tmp/neo_baseline_${tag}.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"wall={d[\\\"generate_wall_s\\\"]:.1f}s tps={d[\\\"prompt_tps\\\"]:.0f} req/s={d[\\\"req_per_s\\\"]:.2f}\")')"
    echo "============================================================"
}

run_scenario 5000 "5000_5050"   # A
run_scenario 1000 "1000_5050"   # B
run_scenario 500  "500_5050"    # C

echo "============================================================"
echo "ALL CHAIN COMPLETE"
echo "============================================================"
for tag in 5000_5050 1000_5050 500_5050; do
    echo "--- $tag ---"
    cat /tmp/neo_baseline_${tag}.json
    echo
done
