#!/usr/bin/env bash
# B1 AVX-512 detok lever — Phase A4 e2e shadow Llama-3.1-8B (TP=2, GPU 4,5)
# run1: VLLM_USE_AVX512_DETOK_INC=0 (native only)
# run2: VLLM_USE_AVX512_DETOK_INC=1 (shadow path 활성)
#
# 사용: bash run.sh native | bash run.sh shadow
set -uo pipefail
MODE="${1:?usage: run.sh native|shadow}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8002
GPUS="4,5"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b1_e2e
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

if [ "$MODE" = "native" ]; then
  FLAG_VAL=0
  OUT_JSON="$POC_DIR/llama8b_native.json"
  OUT_RAW="$POC_DIR/llama8b_native.raw.jsonl"
  BOOT_LOG="$LOGD/boot_native.log"
elif [ "$MODE" = "shadow" ]; then
  FLAG_VAL=1
  OUT_JSON="$POC_DIR/llama8b_shadow.json"
  OUT_RAW="$POC_DIR/llama8b_shadow.raw.jsonl"
  BOOT_LOG="$LOGD/boot_shadow.log"
else
  echo "unknown mode: $MODE"; exit 2
fi

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- boot ---
log "=== B1 e2e $MODE boot (port=$PORT, gpus=$GPUS, flag=$FLAG_VAL) ==="
CUDA_VISIBLE_DEVICES=$GPUS \
  VLLM_USE_AVX512_DETOK_INC=$FLAG_VAL \
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
echo $PID > "$LOGD/${MODE}.pid"
log "PID=$PID  log=$BOOT_LOG"

# --- wait_ready ---
WAIT_READY_MAX=240
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
  tail -50 "$BOOT_LOG"
  exit 1
fi

# --- benchmark: sharegpt 200 prompt × conc=16 × vanilla, stream ---
log "=== bench: sharegpt 200p × conc=16 × vanilla (stream) ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" vllm_config_perf/gating/realistic_eval/throughput_runner.py \
    --in "$PARQ" \
    --method vanilla \
    --model "$MODEL" \
    --model-tag "Llama-3.1-8B-Instruct" \
    --port "$PORT" \
    --max-tokens 8192 \
    --concurrency 16 \
    --limit 200 \
    --corpus sharegpt \
    --out "$OUT_JSON" \
    --raw "$OUT_RAW" \
  2>&1 | tee -a "$LOGD/bench_${MODE}.log"

log "=== bench done → $OUT_JSON ==="

# --- kill backend ---
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null

# orphan VLLM::Worker on GPU 4,5
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F',' '{print $1}' | sort -u); do
  if [ -n "$orphan_pid" ]; then
    # only kill if cmdline mentions VLLM or python serve
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "kill orphan $orphan_pid: $cmd"
      kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done

# --- wait GPU 4,5 free ---
for i in $(seq 1 30); do
  used45=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F',' '$1==4||$1==5 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  if [ "${used45:-1}" -lt 1000 ]; then
    log "GPU 4,5 freed (used=${used45} MiB)"
    break
  fi
  sleep 3
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' '$1==4||$1==5' | tee "$LOGD/${MODE}.gpu_after.txt"

log "=== DONE $MODE ==="
