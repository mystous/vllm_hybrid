#!/usr/bin/env bash
# Correctness check — boot engine twice (JF off, JF on), run 10-prompt
# JSON-schema-constrained inference each time, compare outputs.
set -uo pipefail

cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8007
GPUS="6,7"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b2_jump_forward
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

boot_engine() {
  local jf_flag=$1
  local tag=$2
  local boot_log="$LOGD/cboot_${tag}.log"
  USED67=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F',' '$1==6||$1==7 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  if [ "${USED67:-0}" -gt 4000 ]; then
    log "ABORT — GPU 6,7 busy: ${USED67} MiB"
    exit 1
  fi
  log "=== boot $tag (JF=$jf_flag) ==="
  CUDA_VISIBLE_DEVICES=$GPUS \
    VLLM_USE_XGRAMMAR_JUMP_FORWARD=$jf_flag \
    ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid "$VBIN" serve "$MODEL" \
      --tensor-parallel-size 2 --port "$PORT" \
      --gpu-memory-utilization 0.85 \
      --max-model-len 16384 \
      --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
      --allow-deprecated-quantization \
      > "$boot_log" 2>&1 < /dev/null &
  PID=$!
  echo $PID > "$LOGD/cengine_${tag}.pid"
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      log "READY"
      return 0
    fi
    sleep 1
  done
  log "TIMEOUT"
  return 1
}

kill_engine() {
  local tag=$1
  local PID=$(cat "$LOGD/cengine_${tag}.pid" 2>/dev/null)
  log "kill pid=$PID"
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  [ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null
  kill -9 "$PID" 2>/dev/null
  sleep 3
  for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
    cmd=$(cat /proc/$op/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      kill -9 "$op" 2>/dev/null
    fi
  done
  for i in $(seq 1 20); do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' '$1==6||$1==7 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
    [ "${u:-1}" -lt 1000 ] && { log "GPU 6,7 freed"; break; }
    sleep 2
  done
}

# ---- JF OFF ----
boot_engine 0 jfoff_corr || exit 1
"$PY" "$POC_DIR/correctness_check.py" \
  --in "$PARQ" --port "$PORT" --model "$MODEL" --limit 10 \
  --label off --out "$POC_DIR/correctness_off.json"
kill_engine jfoff_corr

# ---- JF ON ----
boot_engine 1 jfon_corr || exit 1
"$PY" "$POC_DIR/correctness_check.py" \
  --in "$PARQ" --port "$PORT" --model "$MODEL" --limit 10 \
  --label on --out "$POC_DIR/correctness_on.json"
grep -E "xgrammar jump-forward|JF" "$LOGD/cboot_jfon_corr.log" | tail -10 \
  > "$LOGD/jf_telemetry_correctness.txt" || true
kill_engine jfon_corr

# ---- diff ----
"$PY" "$POC_DIR/compare_correctness.py" \
  --off "$POC_DIR/correctness_off.json" \
  --on "$POC_DIR/correctness_on.json" \
  --out "$POC_DIR/correctness_diff.json" | tee "$LOGD/correctness_diff.log"

log "=== correctness DONE ==="
