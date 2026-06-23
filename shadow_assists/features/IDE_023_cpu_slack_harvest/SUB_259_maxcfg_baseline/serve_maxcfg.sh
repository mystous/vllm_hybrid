#!/usr/bin/env bash
# Max-Config Baseline (모든 성능 플래그 ON) — 매크로 플랜 Phase A.
# 인자: [model] [port] [tier: core|max]
#   core = EP8 + 안전한 성능 env (부팅 확인용)
#   max  = core + 공격적 플래그(flashinfer AR/MoE, trtllm attn, autotune)
set -e
MODEL=${1:-deepseek-ai/DeepSeek-R1}
PORT=${2:-8100}
TIER=${3:-max}
PY=/home/mystous/vllm_dev_prj/bin/python
export HF_HOME=/raid/hf_cache
export VLLM_LOGGING_LEVEL=INFO

# --- 공통 성능 ENV (기본 ON이지만 명시) ---
export VLLM_USE_DEEP_GEMM=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=1
export VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1

EXTRA=""
if [ "$TIER" = "max" ]; then
  export VLLM_ALLREDUCE_USE_FLASHINFER=1
  export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
  # NOTE: R1/EXAONE은 block-wise FP8(128x128) → FLASHINFER_CUTLASS MoE 미지원.
  # MoE 백엔드는 auto(deep_gemm)에 맡긴다. VLLM_USE_FLASHINFER_MOE_FP8 강제 금지.
  EXTRA="--enable-flashinfer-autotune --performance-mode throughput"
fi

echo "[maxcfg] model=$MODEL port=$PORT tier=$TIER"
echo "[maxcfg] ENV: DEEP_GEMM=$VLLM_USE_DEEP_GEMM SYMM_MEM=$VLLM_ALLREDUCE_USE_SYMM_MEM FI_AR=${VLLM_ALLREDUCE_USE_FLASHINFER:-0} FI_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8:-0}"

exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 512 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --optimization-level 3 \
  --port "$PORT" \
  $EXTRA
