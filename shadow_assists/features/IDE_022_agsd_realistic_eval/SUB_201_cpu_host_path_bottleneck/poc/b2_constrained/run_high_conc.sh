#!/usr/bin/env bash
# B2 — replicate at higher concurrency (64) to expose bitmask scaling.
set -uo pipefail

cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8007
GPUS="6,7"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b2_constrained
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

USED67=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$1==6||$1==7 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
if [ "${USED67:-0}" -gt 4000 ]; then
  log "ABORT — GPU 6,7 busy: used=${USED67} MiB"; exit 1
fi
log "pre-check OK"

BOOT_LOG="$LOGD/boot_hc.log"
CUDA_VISIBLE_DEVICES=$GPUS \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 2 --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --allow-deprecated-quantization \
    > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > "$LOGD/engine_hc.pid"

for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "READY in ${i}s"; break
  fi
  sleep 1
done

for MODE in baseline json_schema grammar; do
  OUT_JSON="$POC_DIR/llama8b_${MODE}_hc64.json"
  OUT_RAW="$POC_DIR/llama8b_${MODE}_hc64.raw.jsonl"
  : > "$OUT_RAW"
  log "=== bench HC64: mode=$MODE 500p conc=64 max_tokens=512 ==="
  PYTHONPATH=/workspace/host_vllm_hybrid \
    "$PY" "$POC_DIR/constrained_runner.py" \
      --in "$PARQ" --mode "$MODE" --model "$MODEL" \
      --model-tag "Llama-3.1-8B-Instruct" --port "$PORT" \
      --max-tokens 512 --concurrency 64 --limit 500 --corpus sharegpt \
      --out "$OUT_JSON" --raw "$OUT_RAW" \
    2>&1 | tee -a "$LOGD/bench_${MODE}_hc.log"
done

log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
[ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null
kill -9 "$PID" 2>/dev/null
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u); do
  if [ -n "$orphan_pid" ]; then
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "kill orphan $orphan_pid: $cmd"
      kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done
for i in $(seq 1 30); do
  used67=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1==6||$1==7 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  [ "${used67:-1}" -lt 1000 ] && { log "GPU 6,7 freed"; break; }
  sleep 3
done
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | awk -F',' '$1==6||$1==7' | tee "$LOGD/gpu_after_hc.txt"
log "=== B2 HC DONE ==="
