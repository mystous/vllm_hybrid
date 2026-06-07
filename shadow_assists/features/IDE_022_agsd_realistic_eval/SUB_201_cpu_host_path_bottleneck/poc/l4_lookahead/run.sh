#!/usr/bin/env bash
# SUB_201/L4 — ngram global dict 측정
# Qwen2.5-7B-Instruct TP=1, GPU 2, port 8004.
#
# 3 modes (sharegpt 200p × conc=16 × max-tok 256):
#   A_vanilla     : speculative-config OFF
#   B_ngram       : ngram prompt-only (current vLLM behavior)
#   C_ngram_glb   : ngram + global dict (VLLM_NGRAM_GLOBAL_DICT=1)
#
# Usage: run.sh <MODE>
#   ex: run.sh A_vanilla → qwen7b_A_vanilla.json
set -uo pipefail
MODE="${1:?usage: run.sh <MODE>}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8014
GPU="${L4_GPU:-3}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
TAG_MODEL="Qwen2.5-7B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l4_lookahead
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

# --- SPEC config + env flags per mode ---
SPEC_FLAG=""
GLOBAL_DICT=0
case "$MODE" in
  A_vanilla)
    SPEC_FLAG=""
    GLOBAL_DICT=0
    ;;
  B_ngram)
    SPEC_FLAG='--speculative-config {"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'
    GLOBAL_DICT=0
    ;;
  C_ngram_glb)
    SPEC_FLAG='--speculative-config {"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'
    GLOBAL_DICT=1
    ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac

TAG="${MODE}"
OUT_JSON="$POC_DIR/qwen7b_${TAG}.json"
OUT_RAW="$POC_DIR/qwen7b_${TAG}.raw.jsonl"
BOOT_LOG="$LOGD/boot_${TAG}.log"
BENCH_LOG="$LOGD/bench_${TAG}.log"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- safety: GPU $GPU must be free ---
USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F',' -v g="$GPU" '$1==g {gsub(/ /,"",$2); print $2+0; exit}')
if [ "${USED:-0}" -gt 4000 ]; then
  log "ABORT — GPU $GPU already busy: used=${USED} MiB > 4000"
  exit 1
fi
log "pre-check OK — GPU $GPU free (used=${USED} MiB)"

# --- boot ---
log "=== boot $TAG (port=$PORT, gpu=$GPU, glb_dict=$GLOBAL_DICT) ==="
log "SPEC_FLAG=$SPEC_FLAG"

# build cmd as array to handle the speculative-config JSON correctly
BOOT_CMD=(
  "$VBIN" serve "$MODEL"
  --tensor-parallel-size 1
  --port "$PORT"
  --gpu-memory-utilization 0.80
  --max-model-len 16384
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
)
if [ -n "$SPEC_FLAG" ]; then
  BOOT_CMD+=(--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}')
fi

CUDA_VISIBLE_DEVICES=$GPU \
  VLLM_NGRAM_GLOBAL_DICT=$GLOBAL_DICT \
  VLLM_NGRAM_GLOBAL_DICT_MAX=200000 \
  VLLM_NGRAM_NUM_THREADS_CAP=8 \
  VLLM_NGRAM_DIVIDE_BY_TP=0 \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "${BOOT_CMD[@]}" > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > "$LOGD/${TAG}.pid"
log "PID=$PID  log=$BOOT_LOG"

# --- wait_ready ---
WAIT_READY_MAX=300
T_START=$(date +%s)
READY=0
for i in $(seq 1 $WAIT_READY_MAX); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    BOOT_SEC=$(( $(date +%s) - T_START ))
    log "READY in ${BOOT_SEC}s"
    echo "$BOOT_SEC" > "$LOGD/${TAG}.boot_sec"
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  log "TIMEOUT after ${WAIT_READY_MAX}s — abort"
  tail -80 "$BOOT_LOG"
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  [ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null
  kill -9 "$PID" 2>/dev/null
  exit 1
fi

# --- bench ---
log "=== bench: sharegpt 200p × conc=16 × vanilla method × max-tok 256 ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" vllm_config_perf/gating/realistic_eval/throughput_runner.py \
    --in "$PARQ" \
    --method vanilla \
    --model "$MODEL" \
    --model-tag "$TAG_MODEL" \
    --port "$PORT" \
    --max-tokens 256 \
    --concurrency 16 \
    --limit 200 \
    --corpus sharegpt \
    --out "$OUT_JSON" \
    --raw "$OUT_RAW" \
  2>&1 | tee -a "$BENCH_LOG"

log "=== bench done → $OUT_JSON ==="

# --- kill backend ---
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null

# orphan VLLM::Worker
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

# --- wait GPU $GPU free ---
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' -v g="$GPU" '$1==g {gsub(/ /,"",$2); print $2+0; exit}')
  if [ "${used:-1}" -lt 1000 ]; then
    log "GPU $GPU freed (used=${used} MiB)"
    break
  fi
  sleep 3
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' -v g="$GPU" '$1==g' | tee "$LOGD/${TAG}.gpu_after.txt"

log "=== DONE $TAG ==="
