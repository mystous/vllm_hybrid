#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wrapper for eval/run_neo_e2e_smoke.py — captures the merged
# stdout/stderr to a log file so the in-process verification can
# grep for NEO gate messages emitted from the EngineCore subprocess.
set -euo pipefail

LOG_FILE="${LOG_FILE:-/tmp/neo_smoke.log}"
PY="${PY:-/workspace/vllm_dev_prj/bin/python}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

: > "$LOG_FILE"

HF_HUB_OFFLINE=1 LD_PRELOAD=/usr/lib64/libcuda.so.1 \
    "$PY" -u "$SCRIPT_DIR/run_neo_e2e_smoke.py" \
        --log-file "$LOG_FILE" \
        2>&1 | tee -a "$LOG_FILE"
