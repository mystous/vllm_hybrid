#!/usr/bin/env bash
# H-1: model-weight FP8 quantization. Three candidate models, 5-sweep each.
#
#  H1A : RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic  (compressed-tensors w8a8, dynamic act)
#  H1B : neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8       (per-channel w + per-token act)
#  H1C : meta-llama/Llama-3.1-8B-Instruct + --quantization fp8 (in-flight dynamic quant)
#
# Same workload as baseline. Compares 5-sweep mean vs hw_heavy_baseline tps.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export PORT=8092
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=hwh1
source "$ROOT/../hw_heavy_baseline/scripts/lib_heavy.sh"

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

# --- H1A: RedHat dynamic FP8 ----------------------------------------------------
export MODEL="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
EXTRA_ENV=()
EXTRA_CLI=()
do_case_nsweep "h1a_redhat_fp8_dynamic" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

# --- H1B: Neural Magic static FP8 ----------------------------------------------
export MODEL="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
EXTRA_ENV=()
EXTRA_CLI=()
do_case_nsweep "h1b_neuralmagic_fp8" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

# --- H1C: in-flight quantization fp8 -------------------------------------------
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
EXTRA_ENV=()
EXTRA_CLI=( --quantization fp8 )
do_case_nsweep "h1c_inflight_fp8" "$N_SWEEPS"

echo ""
echo "==== H-1 done @ $(date -u +%FT%TZ) ===="
