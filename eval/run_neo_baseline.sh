#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

LOG_FILE="${LOG_FILE:-/tmp/neo_baseline.log}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/neo_baseline.json}"
PY="${PY:-/workspace/vllm_dev_prj/bin/python}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

: > "$LOG_FILE"

cd "$ROOT_DIR"
HF_HUB_OFFLINE=1 LD_PRELOAD=/usr/lib64/libcuda.so.1 \
    "$PY" -u "$SCRIPT_DIR/run_neo_baseline.py" \
        --log-file "$LOG_FILE" \
        --output-file "$OUTPUT_FILE" \
        "$@" \
        2>&1 | tee -a "$LOG_FILE"
