#!/usr/bin/env bash
# kill_engine.sh — stop the vLLM serve started by boot_smoke.sh / run_e2e.sh
# Strategy (CLAUDE.md hazards):
#   - PID/setsid pgroup kill (no self-pkill)
#   - orphan TP worker (VLLM::Worker) → nvidia-smi compute-apps PID 직접 kill
#   - wait for GPU 0-7 free (memory.used <= 500 MiB on each)
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/a1_vllm_integration
LOGD=$ROOT/_logs

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# (1) kill all PID files we know about
for pf in "$LOGD"/*.pid; do
  [ -f "$pf" ] || continue
  pid=$(cat "$pf" 2>/dev/null || true)
  [ -z "$pid" ] && continue
  if [ -d "/proc/$pid" ]; then
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    log "kill pid=$pid pgid=$pgid (from $pf)"
    [ -n "$pgid" ] && kill -TERM -- -"$pgid" 2>/dev/null || true
    # poll up to 30s
    for i in $(seq 1 30); do
      [ -d "/proc/$pid" ] || break
      sleep 1
    done
    if [ -d "/proc/$pid" ]; then
      log "  still alive — SIGKILL"
      [ -n "$pgid" ] && kill -KILL -- -"$pgid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pf"
done

# (2) orphan TP workers via nvidia-smi compute-apps
opids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
for op in $opids; do
  if [ -d "/proc/$op" ]; then
    log "kill orphan compute-app pid=$op"
    kill -KILL "$op" 2>/dev/null || true
  fi
done

# (3) wait for GPU 0-7 free
log "wait for GPU 0-7 free…"
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
  if [ "$busy" -eq 0 ]; then
    log "all 8 GPU free"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
  fi
  sleep 2
done
log "TIMEOUT: GPU still busy"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
exit 1
