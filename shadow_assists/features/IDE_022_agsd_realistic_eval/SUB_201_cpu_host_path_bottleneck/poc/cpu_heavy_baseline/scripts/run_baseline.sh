#!/usr/bin/env bash
# Re-measure baseline: vanilla Llama-3.1-8B-Instruct, TP=8, B3 FaP + L2 + L10.
# Adds mpstat CPU monitoring (parsed via lib_cpu_heavy.sh).
# Single config, 3 sweeps (we already have a 5-sweep hw_heavy_baseline result;
# this just re-validates the harness + measures CPU util with mpstat).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8101
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=3
export EXTRA_LOG_TAG=cpuhBL
source "$SCRIPT_DIR/lib_cpu_heavy.sh"

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

# baseline = vanilla + B3 FaP + L2 prefetch + L10 burst-aware
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' )
do_case_nsweep "baseline_b3l2l10" "$N_SWEEPS"

summarize_tags "baseline_b3l2l10"

echo "==== cpu_heavy_baseline done @ $(date -u +%FT%TZ) ===="
