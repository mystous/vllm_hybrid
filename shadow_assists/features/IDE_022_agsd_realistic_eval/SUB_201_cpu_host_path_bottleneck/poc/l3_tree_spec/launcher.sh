#!/usr/bin/env bash
# L3 — CPU tree spec-decoding launcher (GPU 3, Qwen2.5-32B TP=1)
#
#   up [baseline|tree]     start server on GPU 3, port 8003
#   down                   stop server (PID file in $LOG_DIR)
#   status                 PID liveness
#
# baseline = suffix K=7, single-path (upstream behaviour).
# tree     = suffix K=7, K-branch tree (VLLM_L3_TREE_SPEC=1, VLLM_L3_TREE_BRANCHES=4)
#
# Why GPU 3 only: the L3 task narrows the device count to isolate the
# proposer signal from concurrent runs on other GPUs.

set -euo pipefail
MODE="${2:-baseline}"
MODEL="${L3_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
GMU="${L3_GMU:-0.85}"
MAX_MODEL_LEN="${L3_MAX_MODEL_LEN:-8192}"
PORT="${L3_PORT:-8003}"
VLLM_BIN="${L3_VLLM_BIN:-/workspace/vllm_dev_prj/bin/vllm}"
LOG_DIR="${L3_LOG_DIR:-/tmp/l3_tree_spec_logs}"
NUM_SPEC="${L3_NUM_SPEC:-7}"
mkdir -p "$LOG_DIR"

# Force editable vLLM (host_vllm_hybrid path) — the venv has its own copy in
# site-packages, but our patched suffix_decoding.py lives in the repo tree.
# `vllm.pth` should already pin to the host tree per b200-vllm-build memo.
export PYTHONPATH="/workspace/host_vllm_hybrid:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""

STATS_DIR="${L3_STATS_DIR:-$LOG_DIR/stats}"
mkdir -p "$STATS_DIR"

# Both env vars *and* /tmp/vllm_l3_*.flag files — the launcher uses the
# latter because vLLM's EngineCore subprocess starts with `spawn`, which
# does NOT propagate env vars from the API server.  See the patched
# suffix_decoding.py for the lazy reader.
write_flag() { echo "$2" > "/tmp/vllm_l3_$1.flag"; }
clear_flag() { rm -f "/tmp/vllm_l3_$1.flag"; }

case "$MODE" in
  baseline)
    unset VLLM_L3_TREE_SPEC || true
    unset VLLM_L3_TREE_BRANCHES || true
    clear_flag VLLM_L3_TREE_SPEC
    clear_flag VLLM_L3_TREE_BRANCHES
    TAG="baseline_single_K${NUM_SPEC}"
    export VLLM_L3_TREE_STATS_PATH="$STATS_DIR/baseline.jsonl"
    : > "$VLLM_L3_TREE_STATS_PATH"   # truncate so each up = clean record
    write_flag VLLM_L3_TREE_STATS_PATH "$VLLM_L3_TREE_STATS_PATH"
    ;;
  tree)
    export VLLM_L3_TREE_SPEC=1
    export VLLM_L3_TREE_BRANCHES="${L3_TREE_BRANCHES:-4}"
    write_flag VLLM_L3_TREE_SPEC 1
    write_flag VLLM_L3_TREE_BRANCHES "$VLLM_L3_TREE_BRANCHES"
    TAG="tree_b${VLLM_L3_TREE_BRANCHES}_K${NUM_SPEC}"
    export VLLM_L3_TREE_STATS_PATH="$STATS_DIR/tree_b${VLLM_L3_TREE_BRANCHES}.jsonl"
    : > "$VLLM_L3_TREE_STATS_PATH"
    write_flag VLLM_L3_TREE_STATS_PATH "$VLLM_L3_TREE_STATS_PATH"
    ;;
  *)
    echo "unknown mode: $MODE (expected baseline|tree)" >&2
    exit 1 ;;
esac

start() {
  echo "[l3/$MODE] starting on GPU 3, port $PORT (mode=$TAG, K=$NUM_SPEC, gmu=$GMU)..."
  CUDA_VISIBLE_DEVICES=3 nohup "$VLLM_BIN" serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_MODEL_LEN" \
    --speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":${NUM_SPEC}}" \
    > "$LOG_DIR/server_${MODE}.log" 2>&1 &
  echo $! > "$LOG_DIR/.pid_server"
  echo "[l3/$MODE] pid=$(cat "$LOG_DIR/.pid_server")  log=$LOG_DIR/server_${MODE}.log"
}

stop() {
  # PID-based stop; never `pkill -f vllm serve` (self-match risk).
  for f in "$LOG_DIR"/.pid_*; do
    [ -e "$f" ] || continue
    pid=$(cat "$f"); pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    if [ -n "$pgid" ]; then
      kill -TERM -"$pgid" 2>/dev/null || true
    fi
    kill -TERM "$pid" 2>/dev/null || true
    rm -f "$f"
  done
  # Reap any orphan VLLM::Worker on GPU 3.
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
    [ -z "$p" ] && continue
    # only kill if running on GPU 3
    if grep -q "CUDA_VISIBLE_DEVICES=3" "/proc/$p/environ" 2>/dev/null; then
      kill -TERM "$p" 2>/dev/null || true
    fi
  done
  echo "[l3] stop signaled"
}

wait_ready() {
  url="http://127.0.0.1:$PORT"
  for i in $(seq 1 180); do
    if curl -sf "$url/v1/models" >/dev/null 2>&1; then
      echo "[l3] backend READY after $((i*5))s"
      return 0
    fi
    sleep 5
  done
  echo "[l3] TIMEOUT waiting for backend" >&2
  return 1
}

case "${1:-up}" in
  up)      start; wait_ready ;;
  down)    stop ;;
  status)
    for f in "$LOG_DIR"/.pid_*; do
      [ -e "$f" ] || continue
      n=$(basename "$f" | sed 's/^.pid_//'); pid=$(cat "$f")
      if kill -0 "$pid" 2>/dev/null; then echo "[status] $n pid=$pid RUNNING"
      else echo "[status] $n pid=$pid DEAD"; fi
    done ;;
  *) echo "usage: $0 {up|down|status} [baseline|tree]"; exit 1 ;;
esac
