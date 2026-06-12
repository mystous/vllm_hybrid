#!/usr/bin/env bash
# Baseline re-measurement: vanilla Llama-3.1-8B-Instruct, TP=8, B3 FaP + L2 + L10.
# Matches the protocol used by hw_custom_round_3 (KV fp8 was already on there, but the
# "ground truth" baseline cited in the brief is fp16 KV vanilla = 22,077.9 tps). We
# measure BOTH:
#   - vanilla_fp16kv : pristine baseline (no kv-cache-dtype, no compilation overrides)
#   - vanilla_b3l2l10: prior "baseline + recommended levers" (FaP + L2 + L10)
# This anchors H-1..H-5 deltas against a freshly re-validated number.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8091
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=hwhB
source "$SCRIPT_DIR/lib_heavy.sh"

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

# --- Case A: pristine vanilla (fp16 KV, no compilation override) -----------------
EXTRA_ENV=()
EXTRA_CLI=()
do_case_nsweep "vanilla_fp16kv" "$N_SWEEPS"

sleep 5; wait_gpu_free || true

# --- Case B: vanilla + B3 FaP (FULL_AND_PIECEWISE) + L2 prefetch + L10 burst -----
# Matches the "baseline" cited in the brief (22,077.9 ± 151.7 tps).
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' )
do_case_nsweep "vanilla_b3l2l10" "$N_SWEEPS"

echo ""
echo "==== Baseline done @ $(date -u +%FT%TZ) ===="
