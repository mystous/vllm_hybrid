#!/usr/bin/env bash
# C-1: NGram spec decode with CPU heavy multi-thread + precompute.
# Mechanism: ngram_proposer.py already supports parallelization across cores
# via numba (VLLM_NGRAM_NUM_THREADS_CAP) and a background precompute thread
# (VLLM_NGRAM_PRECOMPUTE). These both move scheduler/proposer work to CPU
# while reducing GPU verify count when accept_rate > 0.
#
# Sweeps:
#   C1a: ngram K=3 with default cpu-bound settings (single thread)
#   C1b: ngram K=3 with NGRAM_NUM_THREADS_CAP=32, PRECOMPUTE=1, THRESHOLD=512
#   C1c: ngram K=5 (longer drafts) with same CPU-heavy settings
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8102
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=cpuhC1
# Re-use the harness library from cpu_heavy_baseline.
source /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/cpu_heavy_baseline/scripts/lib_cpu_heavy.sh

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

NGRAM_CFG='{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'

# --- C1a: ngram K=3, default (single CPU thread, no precompute) ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --speculative-config "$NGRAM_CFG" )
do_case_nsweep "C1a_ngram_k3_default" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

# --- C1b: ngram K=3 + cpu-heavy: 32 threads + precompute + lower threshold ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
            VLLM_NGRAM_NUM_THREADS_CAP=32 VLLM_NGRAM_DIVIDE_BY_TP=0
            VLLM_NGRAM_PRECOMPUTE=1 VLLM_NGRAM_THRESHOLD=512 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --speculative-config "$NGRAM_CFG" )
do_case_nsweep "C1b_ngram_k3_cpuheavy" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

# --- C1c: ngram K=5 + cpu-heavy ---
NGRAM_CFG5='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}'
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
            VLLM_NGRAM_NUM_THREADS_CAP=32 VLLM_NGRAM_DIVIDE_BY_TP=0
            VLLM_NGRAM_PRECOMPUTE=1 VLLM_NGRAM_THRESHOLD=512 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --speculative-config "$NGRAM_CFG5" )
do_case_nsweep "C1c_ngram_k5_cpuheavy" "$N_SWEEPS"

summarize_tags "C1a_ngram_k3_default" "C1b_ngram_k3_cpuheavy" "C1c_ngram_k5_cpuheavy"

echo "==== cpu_heavy_C1 done @ $(date -u +%FT%TZ) ===="
