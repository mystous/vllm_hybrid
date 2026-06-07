#!/usr/bin/env bash
# B: SGLang + kt-kernel (AMX INT8) MoE expert CPU offload
set -euo pipefail

MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
KT_WEIGHTS=/workspace/kt_weights/Qwen3-30B-A3B-Instruct-2507-INT8
PYTHON=/workspace/sglang_kt_prj/bin/python
LOG_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/moe_offload
HOST=127.0.0.1
PORT=8001
# Single GPU + CPU AMX (the whole point of expert offload)
TP=${TP:-1}
# Qwen3-30B-A3B: num_experts=128, top-k=8
# Keep small number on GPU (hot path) — try 32 first per LMSYS guide
KT_NUM_GPU_EXPERTS=${KT_NUM_GPU_EXPERTS:-32}
# Total experts across all layers: 48 layers × 128 experts. --kt-num-gpu-experts is per-layer.
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/B_server.log"

echo "[B] launching SGLang kt-kernel offload (TP=$TP, GPU experts/layer=$KT_NUM_GPU_EXPERTS)" | tee "$LOG"
echo "[B] $(date)" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free --format=csv | tee -a "$LOG"

nohup $PYTHON -m sglang.launch_server \
  --host $HOST --port $PORT \
  --model-path "$MODEL_PATH" \
  --tp $TP \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --disable-radix-cache \
  --attention-backend triton \
  --moe-runner-backend triton \
  --disable-cuda-graph \
  --disable-flashinfer-autotune \
  --kt-method BF16 \
  --kt-cpuinfer 112 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts $KT_NUM_GPU_EXPERTS \
  --kt-max-deferred-experts-per-token 2 \
  >> "$LOG" 2>&1 &
echo $! > "$LOG_DIR/B_server.pid"
echo "[B] PID=$(cat $LOG_DIR/B_server.pid) log=$LOG"
