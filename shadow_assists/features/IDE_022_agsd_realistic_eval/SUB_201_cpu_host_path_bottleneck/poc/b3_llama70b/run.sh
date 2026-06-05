#!/usr/bin/env bash
# B3 cudagraph lever — Llama-3.1-70B FaP +30% 재현 검증
#
# 2 sequential runs:
#   A: cudagraph_mode=PIECEWISE     (baseline)
#   B: cudagraph_mode=FULL_AND_PIECEWISE (B3 default)
#
# 조건: TP=4, GPU 0-3, port 8003, sharegpt 200p × conc=32 × max-tokens 8192
#
# 사용: bash run.sh A_piecewise | bash run.sh B_fap
set -uo pipefail
MODE="${1:?usage: run.sh A_piecewise|B_fap}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8003
GPUS="0,1,2,3"
MODEL="meta-llama/Llama-3.1-70B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_llama70b
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

LIB_SO=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/build/libpinned_pool.so

if [ "$MODE" = "A_piecewise" ]; then
  CG_MODE="PIECEWISE"
  OUT_JSON="$POC_DIR/llama70b_A_piecewise.json"
  OUT_RAW="$POC_DIR/llama70b_A_piecewise.raw.jsonl"
  BOOT_LOG="$LOGD/boot_A_piecewise.log"
elif [ "$MODE" = "B_fap" ]; then
  CG_MODE="FULL_AND_PIECEWISE"
  OUT_JSON="$POC_DIR/llama70b_B_fap.json"
  OUT_RAW="$POC_DIR/llama70b_B_fap.raw.jsonl"
  BOOT_LOG="$LOGD/boot_B_fap.log"
else
  echo "unknown mode: $MODE"; exit 2
fi

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- pre-check: GPU 0-3 must be free ---
log "=== pre-check: GPU 0-3 free (< 4000 MiB) ==="
used03=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$1>=0 && $1<=3 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
log "GPU 0-3 used = ${used03} MiB"
if [ "${used03:-99999}" -gt 4000 ]; then
  log "ABORT: GPU 0-3 not free"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  exit 1
fi

# --- boot ---
log "=== B3 cudagraph $MODE boot (port=$PORT, gpus=$GPUS, cg=$CG_MODE) ==="
CUDA_VISIBLE_DEVICES=$GPUS \
  VLLM_KV_TIERING_DRAM=0 \
  VLLM_KV_TIER_TELEMETRY=0 \
  VLLM_KV_TIERING_POOL_LIB="$LIB_SO" \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 4 --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --compilation-config "{\"cudagraph_mode\":\"$CG_MODE\"}" \
    --allow-deprecated-quantization \
    > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
echo $PID > "$LOGD/${MODE}.pid"
log "PID=$PID  log=$BOOT_LOG"

# --- wait_ready ---
WAIT_READY_MAX=900
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
  tail -120 "$BOOT_LOG"
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$PGID" ]; then kill -9 -"$PGID" 2>/dev/null; fi
  kill -9 "$PID" 2>/dev/null
  exit 1
fi

# --- benchmark: sharegpt 200 prompt × conc=32 × vanilla, stream ---
log "=== bench: sharegpt 200p × conc=32 × vanilla (stream) ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" vllm_config_perf/gating/realistic_eval/throughput_runner.py \
    --in "$PARQ" \
    --method vanilla \
    --model "$MODEL" \
    --model-tag "Llama-3.1-70B-Instruct" \
    --port "$PORT" \
    --max-tokens 8192 \
    --concurrency 32 \
    --limit 200 \
    --corpus sharegpt \
    --out "$OUT_JSON" \
    --raw "$OUT_RAW" \
  2>&1 | tee -a "$LOGD/bench_${MODE}.log"

log "=== bench done → $OUT_JSON ==="

# --- kill backend ---
log "=== kill backend pid=$PID (SIGTERM first) ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -TERM -"$PGID" 2>/dev/null
  sleep 3
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null

# --- orphan sweep (only PIDs on GPU 0-3) ---
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F',' '{print $1}' | sort -u); do
  if [ -n "$orphan_pid" ]; then
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "orphan candidate $orphan_pid: $cmd"
      kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done

# --- wait GPU 0-3 free ---
for i in $(seq 1 60); do
  used03=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F',' '$1>=0 && $1<=3 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  if [ "${used03:-1}" -lt 4000 ]; then
    log "GPU 0-3 freed (used=${used03} MiB)"
    break
  fi
  sleep 3
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' '$1>=0 && $1<=3' | tee "$LOGD/${MODE}.gpu_after.txt"

log "=== DONE $MODE ==="
