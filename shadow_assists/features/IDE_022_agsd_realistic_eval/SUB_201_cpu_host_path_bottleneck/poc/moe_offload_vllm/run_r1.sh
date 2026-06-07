#!/bin/bash
# R1 server + measurement. Args: PROFILE PORT TP LABEL
set -uo pipefail
PROFILE=${1:-vanilla}
PORT=${2:-8019}
TP=${3:-8}
LABEL=${4:-default}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lifecycle.sh"

LOG_DIR="$HERE/logs/r1/${LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[r1] profile=$PROFILE port=$PORT tp=$TP label=$LABEL log_dir=$LOG_DIR"

bash "$HERE/start_server_r1.sh" "$PROFILE" "$PORT" "$TP"
PIDFILE="$HERE/logs/r1/${PROFILE}_tp${TP}.pid"

if ! wait_ready "$PORT" 2400; then
    echo "[r1] server failed to become ready"
    kill_pgroup "$PIDFILE"
    for g in 0 1 2 3 4 5 6 7; do kill_gpu_orphans "$g"; done
    exit 1
fi

top -bn1 | head -5 > "$LOG_DIR/cpu_before.txt" 2>&1 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$LOG_DIR/gpu_before.txt" 2>&1 || true

# Smaller test first: 20 prompts conc=8 to confirm sanity
LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib \
    /workspace/vllm_dev_prj/bin/python "$HERE/measure_client.py" \
    --url "http://127.0.0.1:${PORT}/v1" \
    --model r1-offload-test --n 100 --conc 8 --max-tokens 256 \
    --out "$LOG_DIR/result.json" 2>&1 | tee "$LOG_DIR/measure.log"

top -bn1 | head -5 > "$LOG_DIR/cpu_after.txt" 2>&1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_after.txt" 2>&1 || true

kill_pgroup "$PIDFILE"
for g in 0 1 2 3 4 5 6 7; do kill_gpu_orphans "$g"; done
for g in 0 1 2 3 4 5 6 7; do wait_gpu_free "$g" 60 || true; done
echo "[r1] done. results in $LOG_DIR"
