#!/usr/bin/env bash
# =============================================================================
# inspect.sh — Single experiment hardware utilization inspector
#
# Reports CPU pinning, per-NUMA CPU utilization, and GPU utilization.
#
# Usage:
#   ./inspect.sh <result_dir>
#   ./inspect.sh eval/basic/H100x8/20260414_054010_*
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <result_dir>" >&2
    exit 1
fi

exec python3 "${SCRIPT_DIR}/hw_inspect.py" "$@"
