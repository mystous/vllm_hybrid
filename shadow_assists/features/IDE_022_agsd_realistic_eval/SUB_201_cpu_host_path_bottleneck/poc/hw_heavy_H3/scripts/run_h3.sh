#!/usr/bin/env bash
# H-3: Eagle3 speculative decoding atop the vanilla baseline (Llama-3.1-8B-Instruct).
# Two configs sweep num_speculative_tokens 3 vs 5 (Eagle defaults vary; 5 is canonical Eagle3).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8094
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=hwh3
source "$ROOT/../hw_heavy_baseline/scripts/lib_heavy.sh"
trap 'echo "[trap]"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

EAGLE_MODEL=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

# --- H3A: k=3 -----------------------------------------------------------------
EXTRA_ENV=()
EXTRA_CLI=( --speculative-config "{\"method\":\"eagle3\",\"model\":\"$EAGLE_MODEL\",\"num_speculative_tokens\":3}" )
do_case_nsweep "h3a_eagle3_k3" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

# --- H3B: k=5 -----------------------------------------------------------------
EXTRA_ENV=()
EXTRA_CLI=( --speculative-config "{\"method\":\"eagle3\",\"model\":\"$EAGLE_MODEL\",\"num_speculative_tokens\":5}" )
do_case_nsweep "h3b_eagle3_k5" "$N_SWEEPS"

echo ""
echo "==== H-3 done @ $(date -u +%FT%TZ) ===="
