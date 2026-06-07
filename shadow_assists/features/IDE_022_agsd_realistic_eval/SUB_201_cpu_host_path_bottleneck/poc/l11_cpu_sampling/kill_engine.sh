#!/usr/bin/env bash
# L11 kill engine — kill PID + orphan GPU workers (CLAUDE.md hazards).
set -uo pipefail
MODE="${1:?usage: kill_engine.sh baseline|cpu_sampling}"
POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l11_cpu_sampling
PID_FILE="$POC_DIR/_logs/${MODE}.pid"
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
    if [ -n "$PGID" ]; then
      echo "killing pgid $PGID"
      kill -TERM -"$PGID" 2>/dev/null || true
      sleep 3
      kill -KILL -"$PGID" 2>/dev/null || true
    fi
  fi
fi
# nuke orphan workers on GPU 7
ORPHAN=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | awk -F, '{print $1}' | tr -d ' ')
for p in $ORPHAN; do
  CMD=$(ps -p "$p" -o cmd= 2>/dev/null || true)
  if echo "$CMD" | grep -qE "VLLM::Worker|vllm serve"; then
    echo "killing orphan worker pid $p"
    kill -KILL "$p" 2>/dev/null || true
  fi
done
sleep 2
echo "kill done"
