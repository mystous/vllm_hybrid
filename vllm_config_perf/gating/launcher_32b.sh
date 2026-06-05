#!/usr/bin/env bash
# AGSD end-to-end launcher (Qwen 32B / TP=4×2 / DGX B200) — SUB run.
#
#   GPU 0-3 : vLLM serve  Qwen 32B (vanilla, port 8001, gmu=0.85)
#   GPU 4-7 : vLLM serve  Qwen 32B (suffix+FULL_AND_PIECEWISE, port 8002, gmu=0.80)  # B200 default (b3_sched DECISION)
#   CPU     : agsd_router          (FastAPI, port 8000)
#
# Adapted from vllm_config_perf/gating/launcher.sh:
#   - model = Qwen/Qwen2.5-32B-Instruct, TP=4 (was 7B / TP=1)
#   - vanilla on CUDA 0-3, trident on CUDA 4-7 (was single GPU 1 / 2)
#   - max-model-len 16640 (8192 in + 8192 out + slack)
#   - VLLM bin = /workspace/vllm_dev_prj/bin/vllm (no repo .venv here)

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${AGSD_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
GMU_VANILLA="${AGSD_GMU_VANILLA:-0.85}"
GMU_TRIDENT="${AGSD_GMU_TRIDENT:-0.80}"
MAX_MODEL_LEN="${AGSD_MAX_MODEL_LEN:-16640}"
VLLM_BIN="${AGSD_VLLM_BIN:-/workspace/vllm_dev_prj/bin/vllm}"
PYBIN="${AGSD_PYBIN:-/workspace/vllm_dev_prj/bin/python}"
LOG_DIR="${AGSD_LOG_DIR:-/tmp/agsd_32b_logs}"
mkdir -p "$LOG_DIR"

# Trident core env (suffix + FULL_AND_PIECEWISE on B200; b3_sched DECISION)
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SUB_047 ngram thread lever (벤치 자체엔 영향 없음, suffix path 사용)
export VLLM_NGRAM_NUM_THREADS_CAP=8
export VLLM_NGRAM_DIVIDE_BY_TP=0

start_vanilla() {
  echo "[vanilla] starting on GPU 0-3, port 8001 (gmu=$GMU_VANILLA, no spec)..."
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$VLLM_BIN" serve "$MODEL" \
    --tensor-parallel-size 4 \
    --port 8001 \
    --gpu-memory-utilization "$GMU_VANILLA" \
    --max-model-len "$MAX_MODEL_LEN" \
    --disable-log-requests \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    > "$LOG_DIR/vanilla.log" 2>&1 &
  echo $! > "$LOG_DIR/.pid_vanilla"
}

start_trident() {
  echo "[trident] starting on GPU 4-7, port 8002 (suffix K=32, gmu=$GMU_TRIDENT)..."
  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup "$VLLM_BIN" serve "$MODEL" \
    --tensor-parallel-size 4 \
    --port 8002 \
    --gpu-memory-utilization "$GMU_TRIDENT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --disable-log-requests \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
    > "$LOG_DIR/trident.log" 2>&1 &
  echo $! > "$LOG_DIR/.pid_trident"
}

start_router() {
  echo "[router] starting agsd_router on CPU, port 8000..."
  export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1"
  export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1"
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  nohup "$PYBIN" -m uvicorn \
    vllm_config_perf.gating.agsd_router:app \
    --host 0.0.0.0 --port 8000 --loop uvloop --workers 1 \
    > "$LOG_DIR/router.log" 2>&1 &
  echo $! > "$LOG_DIR/.pid_router"
}

wait_for_backend() {
  local url=$1; local name=$2
  echo "[wait] $name at $url ..."
  for i in $(seq 1 240); do
    if curl -sf "$url/v1/models" >/dev/null 2>&1; then
      echo "[wait] $name READY (after $((i*5))s)"
      return 0
    fi
    sleep 5
  done
  echo "[wait] $name TIMEOUT" >&2
  return 1
}

stop_all() {
  for f in "$LOG_DIR"/.pid_*; do
    [ -e "$f" ] || continue
    pid=$(cat "$f"); kill "$pid" 2>/dev/null || true; rm -f "$f"
  done
  pkill -f "vllm serve" 2>/dev/null || true
  echo "[stop] all processes terminated"
}

case "${1:-up}" in
  up)
    start_vanilla
    start_trident
    wait_for_backend "http://127.0.0.1:8001" "vanilla"
    wait_for_backend "http://127.0.0.1:8002" "trident"
    start_router
    sleep 3
    echo ""
    echo "[ready] AGSD 32B stack running."
    echo "  - router  : http://127.0.0.1:8000  (vanilla GPU0-3 + trident GPU4-7)"
    echo "  - logs    : $LOG_DIR"
    ;;
  down)  stop_all ;;
  status)
    for f in "$LOG_DIR"/.pid_*; do
      [ -e "$f" ] || continue
      name=$(basename "$f" | sed 's/^.pid_//'); pid=$(cat "$f")
      if kill -0 "$pid" 2>/dev/null; then echo "[status] $name (pid $pid) RUNNING"; else echo "[status] $name (pid $pid) DEAD"; fi
    done ;;
  *) echo "usage: $0 {up|down|status}"; exit 1 ;;
esac
