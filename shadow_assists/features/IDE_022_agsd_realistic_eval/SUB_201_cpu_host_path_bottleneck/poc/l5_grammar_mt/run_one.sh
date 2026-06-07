#!/usr/bin/env bash
# SUB_201 L5 — CPU multi-thread grammar state advance PoC.
# Boot once on GPU 6 (TP=1), then run JSON-schema constrained decode
# with VLLM_GRAMMAR_MULTITHREAD = {0, 1} back-to-back via reboot per flag
# (the manager caches the executor at __init__ so flipping mid-flight
# would only affect the dispatch path, not the executor creation; we
# reboot to keep the comparison clean).
set -uo pipefail

cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8055
GPU="6"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l5_grammar_mt
B2_RUNNER=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b2_constrained/constrained_runner.py
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- pick mt mode from CLI ---
MT="${1:-0}"   # 0 = baseline (single-thread), 1 = multi-thread grammar
MT_TAG="mt${MT}"
log "=== L5 run: VLLM_GRAMMAR_MULTITHREAD=${MT} tag=${MT_TAG} ==="

# --- safety: GPU 6 must be free ---
USED6=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$1==6 {gsub(/ /,"",$2); print $2+0}')
if [ "${USED6:-0}" -gt 4000 ]; then
  log "ABORT — GPU 6 already busy: used=${USED6} MiB > 4000"
  exit 1
fi
log "pre-check OK — GPU 6 free (used=${USED6} MiB)"

# --- boot ---
log "=== L5 boot mt=${MT} (port=$PORT, gpu=$GPU) ==="
BOOT_LOG="$LOGD/boot_${MT_TAG}.log"
CUDA_VISIBLE_DEVICES=$GPU \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_GRAMMAR_MULTITHREAD=$MT \
  VLLM_GRAMMAR_MT_MIN_BATCH=4 \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 1 --port "$PORT" \
    --gpu-memory-utilization 0.50 \
    --max-model-len 4096 \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --enforce-eager \
    --allow-deprecated-quantization \
    > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > "$LOGD/engine_${MT_TAG}.pid"
log "PID=$PID  log=$BOOT_LOG"

WAIT_READY_MAX=300
T_START=$(date +%s)
READY=0
for i in $(seq 1 $WAIT_READY_MAX); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    BOOT_SEC=$(( $(date +%s) - T_START ))
    log "READY in ${BOOT_SEC}s"
    echo "$BOOT_SEC" > "$LOGD/boot_${MT_TAG}_sec"
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  log "TIMEOUT after ${WAIT_READY_MAX}s — abort"
  tail -60 "$BOOT_LOG"
  exit 1
fi

# --- runs: json_schema mode only (B2 protocol but max_tokens=256) ---
for MODE in baseline json_schema; do
  OUT_JSON="$POC_DIR/llama8b_${MODE}_${MT_TAG}.json"
  OUT_RAW="$POC_DIR/llama8b_${MODE}_${MT_TAG}.raw.jsonl"
  : > "$OUT_RAW"
  log "=== bench: mode=$MODE 200p conc=16 max_tokens=256 (stream) ==="
  PYTHONPATH=/workspace/host_vllm_hybrid \
    "$PY" "$B2_RUNNER" \
      --in "$PARQ" \
      --mode "$MODE" \
      --model "$MODEL" \
      --model-tag "Llama-3.1-8B-Instruct" \
      --port "$PORT" \
      --max-tokens 256 \
      --concurrency 16 \
      --limit 200 \
      --corpus sharegpt \
      --out "$OUT_JSON" \
      --raw "$OUT_RAW" \
    2>&1 | tee -a "$LOGD/bench_${MODE}_${MT_TAG}.log"
  log "=== bench done: $OUT_JSON ==="
done

# --- kill backend ---
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F',' '{print $1}' | sort -u); do
  if [ -n "$orphan_pid" ]; then
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "kill orphan $orphan_pid: $cmd"
      kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done

for i in $(seq 1 30); do
  used6=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F',' '$1==6 {gsub(/ /,"",$2); print $2+0}')
  if [ "${used6:-1}" -lt 1000 ]; then
    log "GPU 6 freed (used=${used6} MiB)"
    break
  fi
  sleep 3
done
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' '$1==6' | tee "$LOGD/gpu_after_${MT_TAG}.txt"

log "=== L5 mt=${MT} DONE ==="
