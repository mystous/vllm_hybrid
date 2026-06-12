#!/usr/bin/env bash
# C-4: stacks of confirmed-positive lever (fp8 KV) with brief's CPU activation
# levers (CPU sampling + CPU-side tokenize/detokenize already in baseline).
# Also attempts max-num-seqs and scheduler-step tuning to see if we can
# extract more out of GPU bound regime.
#
# Sweeps:
#   C4a: fp8 + VLLM_CPU_SAMPLING=1  (stack)
#   C4b: fp8 + max-num-seqs=512 (default 256) — push more concurrency to GPU
#   C4c: fp8 + max-num-seqs=512 + VLLM_CPU_SAMPLING=1 (final stack)
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8106
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=3
export EXTRA_LOG_TAG=cpuhC4
source /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/cpu_heavy_baseline/scripts/lib_cpu_heavy.sh

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

# --- C4a: fp8 + CPU sampling stack ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
            VLLM_CPU_SAMPLING=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --kv-cache-dtype fp8 )
do_case_nsweep "C4a_fp8_cpusample" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

# --- C4b: fp8 + max-num-seqs=512 ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --kv-cache-dtype fp8 --max-num-seqs 512 )
do_case_nsweep "C4b_fp8_seqs512" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

# --- C4c: fp8 + max-num-seqs=512 + CPU sampling ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
            VLLM_CPU_SAMPLING=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --kv-cache-dtype fp8 --max-num-seqs 512 )
do_case_nsweep "C4c_fp8_seqs512_cpusample" "$N_SWEEPS"

summarize_tags "C4a_fp8_cpusample" "C4b_fp8_seqs512" "C4c_fp8_seqs512_cpusample"

echo "==== cpu_heavy_C4 done @ $(date -u +%FT%TZ) ===="
