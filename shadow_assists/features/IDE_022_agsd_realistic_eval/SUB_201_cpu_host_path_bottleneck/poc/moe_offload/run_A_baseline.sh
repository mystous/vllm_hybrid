#!/usr/bin/env bash
# A: full-GPU baseline (SGLang, no kt-kernel options)
# Same backend as B → controls for engine differences.
set -euo pipefail

MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
PYTHON=/workspace/sglang_kt_prj/bin/python
LOG_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/moe_offload
HOST=127.0.0.1
PORT=8000
TP=${TP:-2}  # Qwen3-30B fits in 1 B200(180GB); use TP=2 for headroom + decode parallelism
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/A_server.log"

echo "[A] launching SGLang baseline (TP=$TP) on port $PORT (no kt options)" | tee "$LOG"
echo "[A] $(date)" | tee -a "$LOG"
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
  >> "$LOG" 2>&1 &
echo $! > "$LOG_DIR/A_server.pid"
echo "[A] PID=$(cat $LOG_DIR/A_server.pid) log=$LOG"
