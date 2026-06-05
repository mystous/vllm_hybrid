#!/usr/bin/env bash
# A2 KV DRAM tiering lever — Phase B11 (multi-turn workload, prefix hit ↑)
# Qwen2.5-7B-Instruct (TP=1, GPU 0 only)
#
# 목적:
#   * B10 async-evict 로 evict overlap 회복 했지만 fetch_block=0 (wildchat 은
#     prefix-cache hit 가 낮아 fetch path 가 발화 안 함) → net tps 회복 못함.
#   * B11: 합성 multi-turn corpus (공유 system prompt ~4265 tok × 5 turn × 200 conv,
#     1000 rows, interleaved) 로 prefix-cache hit ↑ → fetch_block > 0 확인.
#   * 두 run: native (lever OFF) vs tier+async (lever ON + async evict).
#
# 사용: bash run_b11_multiturn.sh native | bash run_b11_multiturn.sh tier
set -uo pipefail
MODE="${1:?usage: run_b11_multiturn.sh native|tier}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8006
GPUS="0"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG="Qwen2.5-7B-Instruct"

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/a2_e2e
PARQ="$POC_DIR/multiturn_workload/multiturn_200x5.parquet"
CORPUS=multiturn
CONC=64
MAX_TOKENS=512    # multi-turn 워크로드 본질은 prefix 반복 → gen 짧게 (시간/메모리 budget)
LIMIT=1000        # 200 conv × 5 turn

LOGD="$POC_DIR/_logs_b11_multiturn"
mkdir -p "$LOGD"

LIB_SO=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/build/libpinned_pool.so

if [ "$MODE" = "native" ]; then
  TIER_DRAM=0
  TIER_ASYNC=0
  OUT_JSON="$POC_DIR/qwen7b_b11_native.json"
  OUT_RAW="$POC_DIR/qwen7b_b11_native.raw.jsonl"
  BOOT_LOG="$LOGD/boot_native.log"
elif [ "$MODE" = "tier" ]; then
  TIER_DRAM=1
  TIER_ASYNC=1
  OUT_JSON="$POC_DIR/qwen7b_b11_tier_async.json"
  OUT_RAW="$POC_DIR/qwen7b_b11_tier_async.raw.jsonl"
  BOOT_LOG="$LOGD/boot_tier_async.log"
else
  echo "unknown mode: $MODE"; exit 2
fi

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- pre-flight GPU 0 free check ---
GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
log "Pre-flight: GPU 0 used=${GPU0_USED} MiB"
if [ "${GPU0_USED:-99999}" -gt 4000 ]; then
  log "ABORT: GPU 0 not free (${GPU0_USED} MiB used)"
  exit 1
fi

# --- boot ---
log "=== A2 e2e B11 multiturn $MODE boot (port=$PORT, gpu=$GPUS, DRAM=$TIER_DRAM ASYNC=$TIER_ASYNC) ==="
CUDA_VISIBLE_DEVICES=$GPUS \
  VLLM_KV_TIERING_DRAM=$TIER_DRAM \
  VLLM_KV_TIER_ASYNC=$TIER_ASYNC \
  VLLM_KV_TIER_TELEMETRY=1 \
  VLLM_KV_TIERING_POOL_LIB="$LIB_SO" \
  VLLM_KV_TIERING_DRAM_BYTES=34359738368 \
  VLLM_PINNED_POOL_AUTO_BUDGET=1 \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 1 --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
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

# --- prefix-cache hit baseline (boot 직후) ---
log "=== /metrics prefix baseline ==="
curl -sf "http://127.0.0.1:$PORT/metrics" \
  | grep -E "prefix_cache|gpu_prefix_cache" \
  | tee "$LOGD/${MODE}.prefix_pre.txt" || true

# --- benchmark: multiturn 1000 rows × conc=64 × vanilla (stream), max-tokens=512 ---
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

# --- prefix-cache hit post ---
log "=== /metrics prefix post ==="
curl -sf "http://127.0.0.1:$PORT/metrics" \
  | grep -E "prefix_cache|gpu_prefix_cache" \
  | tee "$LOGD/${MODE}.prefix_post.txt" || true

log "=== KV snapshot (boot log) ==="
grep -E "Prefix cache|GPU KV cache usage|hit rate|usage:" "$BOOT_LOG" | tail -40 | tee "$LOGD/${MODE}.kv_snap.txt" || true

# --- kill backend ---
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -TERM -"$PGID" 2>/dev/null
  sleep 20
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null

# --- telemetry scrape ---
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
