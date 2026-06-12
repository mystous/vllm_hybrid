#!/usr/bin/env bash
# C-2: AMX CPU draft model + GPU verify.
# Mechanism: vllm/v1/spec_decode/cpu_amx.py CpuAmxProposer runs Qwen 2.5-0.5B
# on CPU (bf16, AMX via PyTorch / oneDNN) and produces K draft tokens; GPU
# verifies. CPU activation is intrinsic (CPU does model forward).
#
# Sweeps:
#   C2a: cpu_amx_draft K=3 (toy path - no real model, baseline of dispatch overhead)
#   C2b: cpu_amx_draft K=3 + VLLM_USE_AMX_DRAFT=1 + 32 CPU threads (real Qwen-0.5B)
#   C2c: cpu_amx_draft K=5 + VLLM_USE_AMX_DRAFT=1 + 56 CPU threads
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODEL=meta-llama/Llama-3.1-8B-Instruct
export PORT=8103
export TP=8
export MAX_MODEL_LEN=16384
export CONC=64
export NPROMPT=500
export MAX_TOKENS=2048
export N_SWEEPS=5
export EXTRA_LOG_TAG=cpuhC2
source /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/cpu_heavy_baseline/scripts/lib_cpu_heavy.sh

trap 'echo "[trap] cleanup"; for p in $(ls $RUNS/*.pid 2>/dev/null); do kill_pgroup "$(cat $p)" 2>/dev/null; done; exit 130' INT TERM

AMX_K3='{"method":"cpu_amx_draft","num_speculative_tokens":3}'
AMX_K5='{"method":"cpu_amx_draft","num_speculative_tokens":5}'

# --- C2a: toy mode K=3 (just dispatch wire-up, no real CPU model) ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --speculative-config "$AMX_K3" )
do_case_nsweep "C2a_cpu_amx_toy_k3" 3   # 3 sweeps for the toy (we only want overhead floor)
sleep 5; wait_gpu_free || true

# --- C2b: real Qwen 0.5B CPU draft, K=3, 32 threads ---
EXTRA_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1
            VLLM_USE_AMX_DRAFT=1 VLLM_CPU_DRAFT_THREADS=32
            VLLM_CPU_DRAFT_MAX_CTX=128 )
EXTRA_CLI=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
            --speculative-config "$AMX_K3" )
do_case_nsweep "C2b_cpu_amx_real_k3" "$N_SWEEPS"
sleep 5; wait_gpu_free || true

summarize_tags "C2a_cpu_amx_toy_k3" "C2b_cpu_amx_real_k3"

echo "==== cpu_heavy_C2 done @ $(date -u +%FT%TZ) ===="
