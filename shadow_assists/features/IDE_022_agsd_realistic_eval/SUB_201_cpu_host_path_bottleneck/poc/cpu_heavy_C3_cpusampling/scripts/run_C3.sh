#!/usr/bin/env bash
# C-3 (mapped to brief C-1): CPU sampling offload via VLLM_CPU_SAMPLING=1.
# vllm/v1/sample/sampler.py already implements the CPU sampling path (SUB_201 L11).
# Brief calls this "AMX BF16 sampler" — the existing impl is FP32 CPU softmax/topk
# but the activation pattern (D2H logits → CPU sample → H2D tokens) is identical.
# We measure with mpstat to verify cpu_util rises.
#
# Sweeps:
#   C3a: VLLM_CPU_SAMPLING=1 (FP32 CPU)
#   C3b: VLLM_CPU_SAMPLING=1 + threads tuned
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8105
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=3   # 3 first — gauge feasibility, then expand if positive
export EXTRA_LOG_TAG=cpuhC3
source /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/cpu_heavy_baseline/scripts/lib_cpu_heavy.sh

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

# --- C3a: VLLM_CPU_SAMPLING=1 only ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
            VLLM_CPU_SAMPLING=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' )
do_case_nsweep "C3a_cpu_sampling" "$N_SWEEPS"

summarize_tags "C3a_cpu_sampling"

echo "==== cpu_heavy_C3 done @ $(date -u +%FT%TZ) ===="
