#!/usr/bin/env bash
# AGSD end-to-end launcher — 2 vLLM backend + CPU router 동시 기동.
#
# SUB_094 (2026-05-25) 의 구조 재구성:
#   GPU 1 : vLLM serve  Qwen 7B (vanilla, port 8001)
#   GPU 2 : vLLM serve  Qwen 7B (suffix+PIECEWISE, port 8002)
#   CPU   : agsd_router            (FastAPI, port 8000)
#
# 환경 변수:
#   AGSD_MODEL          (default Qwen/Qwen2.5-7B-Instruct)
#   AGSD_TP             (default 1, large model 영역 multi-GPU 영역 변경)
#   AGSD_GMU_VANILLA    (default 0.85)
#   AGSD_GMU_TRIDENT    (default 0.80)
#   AGSD_VLLM_BIN       (default .venv/bin/vllm)

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${AGSD_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TP="${AGSD_TP:-1}"
GMU_VANILLA="${AGSD_GMU_VANILLA:-0.85}"
GMU_TRIDENT="${AGSD_GMU_TRIDENT:-0.80}"
VLLM_BIN="${AGSD_VLLM_BIN:-.venv/bin/vllm}"
LOG_DIR="${AGSD_LOG_DIR:-/tmp/agsd_logs}"
mkdir -p "$LOG_DIR"

# Trident core env (suffix + PIECEWISE)
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SUB_047 ngram thread lever (선택)
export VLLM_NGRAM_NUM_THREADS_CAP=8
export VLLM_NGRAM_DIVIDE_BY_TP=0

start_vanilla() {
  echo "[vanilla] starting on GPU 1, port 8001 (gmu=$GMU_VANILLA, no spec)..."
  CUDA_VISIBLE_DEVICES=1 nohup "$VLLM_BIN" serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --port 8001 \
    --gpu-memory-utilization "$GMU_VANILLA" \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    > "$LOG_DIR/vanilla.log" 2>&1 &
  echo $! > "$LOG_DIR/.pid_vanilla"
}

start_trident() {
  echo "[trident] starting on GPU 2, port 8002 (suffix K=32, gmu=$GMU_TRIDENT)..."
  CUDA_VISIBLE_DEVICES=2 nohup "$VLLM_BIN" serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --port 8002 \
    --gpu-memory-utilization "$GMU_TRIDENT" \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
    > "$LOG_DIR/trident.log" 2>&1 &
  echo $! > "$LOG_DIR/.pid_trident"
}

start_router() {
  echo "[router] starting agsd_router on CPU, port 8000..."
  export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1"
  export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1"
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  nohup .venv/bin/python -m uvicorn \
    vllm_config_perf.gating.agsd_router:app \
    --host 0.0.0.0 --port 8000 --loop uvloop --workers 1 \
    > "$LOG_DIR/router.log" 2>&1 &
  echo $! > "$LOG_DIR/.pid_router"
}

wait_for_backend() {
  local url=$1
  local name=$2
  echo "[wait] $name at $url ..."
  for i in $(seq 1 120); do
    if curl -sf "$url/health" >/dev/null 2>&1 || curl -sf "$url/v1/models" >/dev/null 2>&1; then
      echo "[wait] $name READY"
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
    pid=$(cat "$f")
    kill "$pid" 2>/dev/null || true
    rm -f "$f"
  done
  echo "[stop] all processes terminated"
}

case "${1:-up}" in
  up)
    start_vanilla
    start_trident
    wait_for_backend "http://127.0.0.1:8001" "vanilla"
    wait_for_backend "http://127.0.0.1:8002" "trident"
    start_router
    echo ""
    echo "[ready] AGSD stack running."
    echo "  - router  : http://127.0.0.1:8000"
    echo "  - vanilla : http://127.0.0.1:8001"
    echo "  - trident : http://127.0.0.1:8002"
    echo "  - logs    : $LOG_DIR"
    ;;
  down)
    stop_all
    ;;
  status)
    for f in "$LOG_DIR"/.pid_*; do
      [ -e "$f" ] || continue
      name=$(basename "$f" .pid_)
      pid=$(cat "$f")
      if kill -0 "$pid" 2>/dev/null; then
        echo "[status] $name (pid $pid) RUNNING"
      else
        echo "[status] $name (pid $pid) DEAD"
      fi
    done
    ;;
  *)
    echo "usage: $0 {up|down|status}"
    exit 1
    ;;
esac
