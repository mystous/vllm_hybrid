#!/usr/bin/env bash
# C-2: fp8 KV (re-confirm previously +4.02% lever) + cpu_util measurement.
# - Re-run the only previously-positive lever (KV fp8) with mpstat-based
#   CPU monitoring so we have a direct head-to-head.
# - This is *not* a CPU-activation lever per the brief, but is the closest
#   confirmed-positive throughput lever and a building block for stacking.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8104
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=cpuhC2fp8
source /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/cpu_heavy_baseline/scripts/lib_cpu_heavy.sh

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

# --- C2_fp8: KV fp8 (baseline + fp8) ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --kv-cache-dtype fp8 )
do_case_nsweep "C2_fp8_kv" "$N_SWEEPS"

summarize_tags "C2_fp8_kv"

echo "==== cpu_heavy_C2_fp8 done @ $(date -u +%FT%TZ) ===="
