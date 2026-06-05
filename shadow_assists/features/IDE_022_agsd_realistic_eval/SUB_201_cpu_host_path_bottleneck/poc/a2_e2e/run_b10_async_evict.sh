#!/usr/bin/env bash
# A2 KV DRAM tiering lever — Phase B10 (async evict overlap)
# Qwen2.5-7B-Instruct (TP=1, GPU 0 only)
#
# 목적:
#   * B9 fix 로 lever 가 ×80 작동 확정 (n_evict 41 070, evict_bytes 37.7 GB)
#     했으나 tps -12.8 % — evict wait=True 가 forward 와 sync block.
#   * B10: VLLM_KV_TIER_ASYNC=1 ON 시 evict_block(wait=False) →
#     dedicated tier stream 에서 D2H 가 forward 와 overlap.
#     correctness: get_new_blocks/_drop 에서 lazy event_sync.
#   * 두 run: sync (B9 동작 = ASYNC=0) vs async (ASYNC=1).
#
# 사용: bash run_b10_async_evict.sh sync | bash run_b10_async_evict.sh async
set -uo pipefail
MODE="${1:?usage: run_b10_async_evict.sh sync|async}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8006
GPUS="0"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG="Qwen2.5-7B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet
CORPUS=wildchat
CONC=64
MAX_TOKENS=8192
LIMIT=200

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/a2_e2e
LOGD="$POC_DIR/_logs_b10_async_evict"
mkdir -p "$LOGD"

LIB_SO=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/build/libpinned_pool.so

if [ "$MODE" = "sync" ]; then
  ASYNC_VAL=0
  OUT_JSON="$POC_DIR/qwen7b_b10_sync.json"
  OUT_RAW="$POC_DIR/qwen7b_b10_sync.raw.jsonl"
  BOOT_LOG="$LOGD/boot_sync.log"
elif [ "$MODE" = "async" ]; then
  ASYNC_VAL=1
  OUT_JSON="$POC_DIR/qwen7b_b10_async.json"
  OUT_RAW="$POC_DIR/qwen7b_b10_async.raw.jsonl"
  BOOT_LOG="$LOGD/boot_async.log"
else
  echo "unknown mode: $MODE"; exit 2
fi

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- pre-flight GPU 0 free check (절대 GPU 1-7 건드리지 말 것) ---
GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
log "Pre-flight: GPU 0 used=${GPU0_USED} MiB"
if [ "${GPU0_USED:-99999}" -gt 4000 ]; then
  log "ABORT: GPU 0 not free (${GPU0_USED} MiB used) — likely held by other agent"
  exit 1
fi

# --- boot ---
log "=== A2 e2e B10 async-evict $MODE boot (port=$PORT, gpu=$GPUS, ASYNC=$ASYNC_VAL) ==="
CUDA_VISIBLE_DEVICES=$GPUS \
  VLLM_KV_TIERING_DRAM=1 \
  VLLM_KV_TIER_ASYNC=$ASYNC_VAL \
  VLLM_KV_TIER_TELEMETRY=1 \
  VLLM_KV_TIERING_POOL_LIB="$LIB_SO" \
  VLLM_KV_TIERING_DRAM_BYTES=34359738368 \
  VLLM_PINNED_POOL_AUTO_BUDGET=1 \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 1 --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > "$LOGD/${MODE}.pid"
log "PID=$PID  log=$BOOT_LOG"

# --- wait_ready ---
WAIT_READY_MAX=600
T_START=$(date +%s)
READY=0
for i in $(seq 1 $WAIT_READY_MAX); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    BOOT_SEC=$(( $(date +%s) - T_START ))
    log "READY in ${BOOT_SEC}s"
    echo "$BOOT_SEC" > "$LOGD/${MODE}.boot_sec"
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  log "TIMEOUT after ${WAIT_READY_MAX}s — abort"
  tail -80 "$BOOT_LOG"
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$PGID" ]; then kill -9 -"$PGID" 2>/dev/null; fi
  kill -9 "$PID" 2>/dev/null
  exit 1
fi

# --- bind log echo ---
log "=== Bind wire-up evidence ==="
grep -E "\[KVDramTier\]|kv_cache" "$BOOT_LOG" | tail -40 | tee "$LOGD/${MODE}.bind.txt" || true

# --- benchmark: wildchat 200p × conc=64 × vanilla, max-tokens=8192 ---
log "=== bench: $CORPUS ${LIMIT}p × conc=${CONC} × vanilla (stream), max-tokens=${MAX_TOKENS} ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" vllm_config_perf/gating/realistic_eval/throughput_runner.py \
    --in "$PARQ" \
    --method vanilla \
    --model "$MODEL" \
    --model-tag "$MODEL_TAG" \
    --port "$PORT" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONC" \
    --limit "$LIMIT" \
    --corpus "$CORPUS" \
    --out "$OUT_JSON" \
    --raw "$OUT_RAW" \
  2>&1 | tee -a "$LOGD/bench_${MODE}.log"

log "=== bench done → $OUT_JSON ==="

log "=== KV snapshot ==="
grep -E "Prefix cache|GPU KV cache usage|hit rate|usage:" "$BOOT_LOG" | tail -20 | tee "$LOGD/${MODE}.kv_snap.txt" || true

# --- kill backend (SIGTERM first for clean atexit) ---
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -TERM -"$PGID" 2>/dev/null
  sleep 20
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null

# --- telemetry scrape (atexit dump → stderr → boot log tail) ---
log "=== Telemetry dump ==="
grep -E "\[KVDramTier" "$BOOT_LOG" | tee "$LOGD/${MODE}.tier_dump.txt" || true

# orphan VLLM::Worker on GPU 0
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits -i 0 2>/dev/null \
                    | awk -F',' '{print $1}' | sort -u); do
  if [ -n "$orphan_pid" ]; then
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "orphan candidate $orphan_pid (gpu0): $cmd"
      kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done

# --- wait GPU 0 free ---
for i in $(seq 1 40); do
  used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [ "${used0:-99999}" -lt 4000 ]; then
    log "GPU 0 freed (used=${used0} MiB)"
    break
  fi
  sleep 3
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' '$1==0' | tee "$LOGD/${MODE}.gpu_after.txt"

log "=== DONE $MODE ==="
