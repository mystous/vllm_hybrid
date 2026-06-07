#!/usr/bin/env bash
# Kill vllm pid (process-group) and any orphan compute-apps on GPU 0,1.
set -uo pipefail
RUN="${1:?usage: kill.sh A|B|C|D|E}"
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/stack_l2_l10_on_suffix_fap
PID_FILE="$ROOT/_logs/${RUN}.pid"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  log "kill pgroup $PID"
  if [ -d "/proc/$PID" ]; then
    kill -TERM -- -"$PID" 2>/dev/null || true
    for i in $(seq 1 30); do
      [ -d "/proc/$PID" ] || break
      sleep 1
    done
    if [ -d "/proc/$PID" ]; then
      kill -KILL -- -"$PID" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi

# orphan compute-apps on the GPU 0,1 only (do not touch other GPUs)
for op in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | awk -F',' '{print $1}' | sort -u); do
  if [ -d "/proc/$op" ]; then
    # check the process is actually on GPU 0 or 1
    GPU_OK=$(cat /proc/$op/environ 2>/dev/null | tr '\0' '\n' | grep -E '^CUDA_VISIBLE_DEVICES=' | head -1 | awk -F= '{print $2}')
    if [ "$GPU_OK" = "0,1" ] || [ "$GPU_OK" = "0" ] || [ "$GPU_OK" = "1" ]; then
      log "kill orphan $op (CUDA_VISIBLE_DEVICES=$GPU_OK)"
      kill -KILL "$op" 2>/dev/null || true
    fi
  fi
done

# wait gpu free (only check 0,1)
for i in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1 == 0 || $1 == 1 {print $2}' | awk '$1>500 {c++} END{print c+0}')
  [ "$USED" -eq 0 ] && { log "GPU 0,1 free"; break; }
  sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1 == 0 || $1 == 1'
