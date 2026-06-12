#!/usr/bin/env bash
# H-4: TOTAL stack — H1 winner (FP8 weight) + KV fp8 + Eagle3 best k.
# Requires:
#   H1_WINNER_MODEL  e.g. RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
#   H3_BEST_K        e.g. 5
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

: "${H1_WINNER_MODEL:?must export winner from H-1}"
: "${H3_BEST_K:?must export best k from H-3}"
export MODEL=$H1_WINNER_MODEL
export PORT=8095
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=hwh4
source "$ROOT/../hw_heavy_baseline/scripts/lib_heavy.sh"
trap 'echo "[trap]"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

EAGLE_MODEL=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

EXTRA_ENV=()
EXTRA_CLI=( --kv-cache-dtype fp8 \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$EAGLE_MODEL\",\"num_speculative_tokens\":$H3_BEST_K}" )
do_case_nsweep "h4_total_stack" "$N_SWEEPS"

echo ""
echo "==== H-4 done @ $(date -u +%FT%TZ) ===="
