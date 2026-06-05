#!/usr/bin/env bash
# A2 KV DRAM tiering lever — Phase B10 cross-process RPC plumbing
# Llama-3.1-70B (TP=4, GPU 0-3)
#
# Phase B10:
#   * 신규: vllm/v1/core/_kv_tier_rpc_proxy.py (engine-side RpcProxyTier)
#   * vllm/v1/worker/gpu_worker.py 에 kv_tier_* RPC handlers 추가
#   * vllm/v1/engine/core.py 에서 VLLM_KV_TIER_RPC_BIND=1 시 proxy attach
#
# 세 모드 비교:
#   * native     : VLLM_KV_TIERING_DRAM=0
#   * tier_norpc : VLLM_KV_TIERING_DRAM=1 VLLM_KV_TIER_RPC_BIND=0 (B6 동작, lever 비활성)
#   * tier_rpc   : VLLM_KV_TIERING_DRAM=1 VLLM_KV_TIER_RPC_BIND=1 (B10 patch, lever 활성)
#
# 사용: bash run_b10.sh native | bash run_b10.sh tier_norpc | bash run_b10.sh tier_rpc
set -uo pipefail
MODE="${1:?usage: run_b10.sh native|tier_norpc|tier_rpc}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8003
GPUS="0,1,2,3"
MODEL="meta-llama/Llama-3.1-70B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/a2_e2e
LOGD="$POC_DIR/_logs_b10"
mkdir -p "$LOGD"

LIB_SO=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/build/libpinned_pool.so

case "$MODE" in
  native)
    FLAG_VAL=0; RPC_VAL=0; TLM_VAL=0
    OUT_JSON="$POC_DIR/llama70b_b10_native.json"
    OUT_RAW="$POC_DIR/llama70b_b10_native.raw.jsonl"
    BOOT_LOG="$LOGD/boot_native.log"
    ;;
  tier_norpc)
    FLAG_VAL=1; RPC_VAL=0; TLM_VAL=1
    OUT_JSON="$POC_DIR/llama70b_b10_tier_norpc.json"
    OUT_RAW="$POC_DIR/llama70b_b10_tier_norpc.raw.jsonl"
    BOOT_LOG="$LOGD/boot_tier_norpc.log"
    ;;
  tier_rpc)
    FLAG_VAL=1; RPC_VAL=1; TLM_VAL=1
    OUT_JSON="$POC_DIR/llama70b_b10_tier_rpc.json"
    OUT_RAW="$POC_DIR/llama70b_b10_tier_rpc.raw.jsonl"
    BOOT_LOG="$LOGD/boot_tier_rpc.log"
    ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- boot ---
log "=== A2 e2e B10 $MODE boot (port=$PORT, gpus=$GPUS, tier=$FLAG_VAL, rpc=$RPC_VAL, telemetry=$TLM_VAL) ==="
CUDA_VISIBLE_DEVICES=$GPUS \
  VLLM_KV_TIERING_DRAM=$FLAG_VAL \
  VLLM_KV_TIER_RPC_BIND=$RPC_VAL \
  VLLM_KV_TIER_TELEMETRY=$TLM_VAL \
  VLLM_KV_TIERING_POOL_LIB="$LIB_SO" \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 4 --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --allow-deprecated-quantization \
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
  tail -120 "$BOOT_LOG"
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$PGID" ]; then kill -9 -"$PGID" 2>/dev/null; fi
  kill -9 "$PID" 2>/dev/null
  exit 1
fi

# --- bind log echo (proof of wire-up) ---
log "=== Bind wire-up evidence (grep boot log) ==="
grep -E "\[KVDramTier" "$BOOT_LOG" | tee "$LOGD/${MODE}.bind.txt" || true

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

# --- kill backend (so atexit dumps telemetry) ---
log "=== kill backend pid=$PID (SIGTERM first for clean atexit) ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -TERM -"$PGID" 2>/dev/null
  sleep 4
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null

# --- telemetry scrape (atexit dump → stderr → boot log tail) ---
log "=== Telemetry dump (grep boot log) ==="
grep -E "\[KVDramTier" "$BOOT_LOG" | tee "$LOGD/${MODE}.tier_dump.txt" || true

# orphan VLLM::Worker on GPU 0-3
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F',' '{print $1}' | sort -u); do
  if [ -n "$orphan_pid" ]; then
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      gpus=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null \
             | awk -F',' -v p="$orphan_pid" '$1==p{print $2}')
      log "orphan candidate $orphan_pid (gpus=$gpus): $cmd"
      kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done

# --- wait GPU 0-3 free ---
for i in $(seq 1 40); do
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
