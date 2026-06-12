#!/usr/bin/env bash
# amx_cpu_draft / boot.sh — Llama-3.1-8B target (TP=8) + Llama-3.2-1B CPU draft
# (KV cache + real prefix). Modes: vanilla | cpu_amx_draft | suffix
set -uo pipefail
MODE="${1:?usage: boot.sh vanilla|cpu_amx_draft|suffix}"

cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT="${PORT:-8005}"
GPUS="0,1,2,3,4,5,6,7"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${TP:-8}"
NSPEC="${NSPEC:-5}"
DRAFT_MODEL="${DRAFT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
DRAFT_THREADS="${DRAFT_THREADS:-16}"
DRAFT_MAX_CTX="${DRAFT_MAX_CTX:-512}"
USE_KV="${USE_KV:-1}"
RANK0_ONLY="${RANK0_ONLY:-1}"

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/amx_cpu_draft
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

TAG="${MODE}_k${NSPEC}_$(basename "$MODEL" | tr ' /' '__')"
BOOT_LOG="$LOGD/boot_${TAG}.log"
PID_FILE="$LOGD/${TAG}.pid"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

EXTRA_ARGS=()
case "$MODE" in
  vanilla)
    AMX_PREFIX=""
    ;;
  cpu_amx_draft)
    EXTRA_ARGS+=(--speculative-config "{\"method\":\"cpu_amx_draft\",\"num_speculative_tokens\":${NSPEC}}")
    AMX_PREFIX="VLLM_USE_AMX_DRAFT=1 VLLM_CPU_DRAFT_USE_AMX=0 VLLM_CPU_DRAFT_THREADS=${DRAFT_THREADS} VLLM_CPU_DRAFT_MODEL=${DRAFT_MODEL} VLLM_CPU_DRAFT_MAX_CTX=${DRAFT_MAX_CTX} VLLM_CPU_DRAFT_USE_KV=${USE_KV} VLLM_CPU_DRAFT_RANK0_ONLY=${RANK0_ONLY}"
    ;;
  suffix)
    EXTRA_ARGS+=(--speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":${NSPEC}}")
    AMX_PREFIX=""
    ;;
  *) echo "unknown mode"; exit 2;;
esac

log "=== boot $MODE on $MODEL TP=$TP port=$PORT NSPEC=$NSPEC ==="
log "draft env: $AMX_PREFIX"

ENV_PREFIX="CUDA_VISIBLE_DEVICES=$GPUS ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $AMX_PREFIX"

CMD="$ENV_PREFIX setsid $VBIN serve $MODEL --tensor-parallel-size $TP --port $PORT --gpu-memory-utilization 0.85 --max-model-len 16384 --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'"
for a in "${EXTRA_ARGS[@]}"; do
  q=$(printf "%q" "$a")
  CMD="$CMD $q"
done

log "CMD: $CMD"
bash -c "$CMD > '$BOOT_LOG' 2>&1 < /dev/null &
echo \$! > '$PID_FILE'"

PID=$(cat "$PID_FILE")
log "PID=$PID  log=$BOOT_LOG"

WAIT_READY_MAX="${WAIT_READY_MAX:-900}"
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
  if ! kill -0 "$PID" 2>/dev/null; then
    log "process died early"
    tail -60 "$BOOT_LOG"
    exit 3
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  log "TIMEOUT after ${WAIT_READY_MAX}s"
  tail -120 "$BOOT_LOG"
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$PGID" ]; then kill -9 -"$PGID" 2>/dev/null; fi
  kill -9 "$PID" 2>/dev/null
  exit 1
fi

# 1-prompt smoke
log "=== smoke ==="
curl -sf "http://127.0.0.1:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0.0}" \
  > "$LOGD/${TAG}.smoke.json" 2>&1 || true
cat "$LOGD/${TAG}.smoke.json"
echo
log "=== DONE boot for $TAG (engine live, port=$PORT) ==="
