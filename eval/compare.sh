#!/usr/bin/env bash
# =============================================================================
# compare.sh — N개 벤치마크 결과 비교
#
# Usage:
#   ./compare.sh <result_dir1> <result_dir2> [result_dir3 ...]
#   ./compare.sh results/20260407_*
#   ./compare.sh results/20260407_* -o /tmp/my_comparison
#
# Examples:
#   ./compare.sh results/20260407_175848_* results/20260407_180242_*
#   ./compare.sh results/20260407_1[5-8]*
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <result_dir1> <result_dir2> [...] [-o output_dir]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 results/run_a results/run_b" >&2
    echo "  $0 results/20260407_*" >&2
    echo "  $0 results/20260407_* -o /tmp/comparison" >&2
    exit 1
fi

exec python3 "${SCRIPT_DIR}/compare.py" "$@"
