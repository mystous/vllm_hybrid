#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# C → B chain — 50:50 input/output baseline at 500 then 1000 prompts.
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
    echo "DONE ${tag}"
    cat /tmp/neo_baseline_${tag}.json
    echo
    echo "============================================================"
}

run_scenario 500  "500_5050"     # C
run_scenario 1000 "1000_5050"    # B

echo "ALL CB CHAIN COMPLETE"
