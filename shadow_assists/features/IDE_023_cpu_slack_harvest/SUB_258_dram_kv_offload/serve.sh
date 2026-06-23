#!/usr/bin/env bash
# KV-constrained 서빙. 인자: <mode: gpuonly|dram> <gpu_blocks> [port]
# GPU KV 블록을 강제로 작게(--num-gpu-blocks-override) → working-set > cache → evict 유도.
set -e
MODE=${1:-gpuonly}; BLOCKS=${2:-1200}; PORT=${3:-8000}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PY=/home/mystous/vllm_dev_prj/bin/python
export HF_HOME=/raid/hf_cache
export VLLM_LOGGING_LEVEL=INFO
EXTRA=""
if [ "$MODE" = "dram" ]; then
  EXTRA="--kv-offloading-size 60 --kv-offloading-backend native --disable-hybrid-kv-cache-manager"
fi
echo "[serve] mode=$MODE blocks=$BLOCKS port=$PORT model=$MODEL extra='$EXTRA'"
exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --num-gpu-blocks-override "$BLOCKS" \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --port "$PORT" \
  $EXTRA
