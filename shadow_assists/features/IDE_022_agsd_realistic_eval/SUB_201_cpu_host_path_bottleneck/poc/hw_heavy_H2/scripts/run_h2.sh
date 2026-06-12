#!/usr/bin/env bash
# H-2: Stack winner FP8-weight model + --kv-cache-dtype fp8.
# Caller MUST export H1_WINNER_MODEL (set after H-1 analysis).
# Example: H1_WINNER_MODEL=RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

: "${H1_WINNER_MODEL:?must export winner from H-1}"
export MODEL=$H1_WINNER_MODEL
export PORT=8093
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=hwh2
source "$ROOT/../hw_heavy_baseline/scripts/lib_heavy.sh"
trap 'echo "[trap]"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

EXTRA_ENV=()
EXTRA_CLI=( --kv-cache-dtype fp8 )
do_case_nsweep "h2_w8a8_kvfp8" "$N_SWEEPS"

echo ""
echo "==== H-2 done @ $(date -u +%FT%TZ) ===="
