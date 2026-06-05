#!/usr/bin/env bash
# A1 boot smoke test — Llama-3.1-8B-Instruct + cpu_amx_draft (TP=8, GPU 0-7)
# 목적: vLLM 가 cpu_amx_draft method 로 boot 되고 1 prompt completion 응답.
set -uo pipefail
MODE="${1:?usage: boot_smoke.sh vanilla|cpu_amx_draft|suffix}"

cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT="${PORT:-8005}"
GPUS="0,1,2,3,4,5,6,7"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${TP:-8}"
POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/a1_vllm_integration
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

# Tag log/pid with model + mode to avoid stomp across multi-model sweeps
TAG="${MODE}_$(basename "$MODEL" | tr ' /' '__')"
BOOT_LOG="$LOGD/boot_${TAG}.log"
PID_FILE="$LOGD/${TAG}.pid"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

EXTRA_ARGS=()
case "$MODE" in
  vanilla)
    AMX_PREFIX=""
    ;;
  cpu_amx_draft)
    EXTRA_ARGS+=(--speculative-config '{"method":"cpu_amx_draft","num_speculative_tokens":7}')
    # Env vars DIRECT-prefix on the serve invocation (bash array hazard 회피)
    AMX_PREFIX="VLLM_USE_AMX_DRAFT=1 VLLM_CPU_DRAFT_USE_AMX=1 VLLM_CPU_DRAFT_THREADS=16"
    ;;
  suffix)
    EXTRA_ARGS+=(--speculative-config '{"method":"suffix","num_speculative_tokens":7}')
    AMX_PREFIX=""
    ;;
  *) echo "unknown mode"; exit 2;;
esac

log "=== boot $MODE on $MODEL TP=$TP port=$PORT ==="

# direct-prefix env vars (CLAUDE.md hazard avoidance)
ENV_PREFIX="CUDA_VISIBLE_DEVICES=$GPUS ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $AMX_PREFIX"

CMD="$ENV_PREFIX setsid $VBIN serve $MODEL --tensor-parallel-size $TP --port $PORT --gpu-memory-utilization 0.85 --max-model-len 16384 --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'"
for a in "${EXTRA_ARGS[@]}"; do
  # quote args with single-quote-escape so bash -c parses correctly
  q=$(printf "%q" "$a")
  CMD="$CMD $q"
done

log "CMD: $CMD"
bash -c "$CMD > '$BOOT_LOG' 2>&1 < /dev/null &
echo \$! > '$PID_FILE'"

PID=$(cat "$PID_FILE")
log "PID=$PID  log=$BOOT_LOG"

# wait_ready
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
  log "TIMEOUT after ${WAIT_READY_MAX}s — abort"
  tail -120 "$BOOT_LOG"
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$PGID" ]; then kill -9 -"$PGID" 2>/dev/null; fi
  kill -9 "$PID" 2>/dev/null
  exit 1
fi

# 1-prompt smoke
log "=== smoke: 1 completion ==="
curl -sf "http://127.0.0.1:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0.0}" \
  > "$LOGD/${TAG}.smoke.json" 2>&1 || true

log "=== smoke response: ==="
cat "$LOGD/${TAG}.smoke.json"
echo
log "=== check spec metrics (if any) ==="
curl -sf "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
  | grep -E "spec_decode|cpu_amx|num_drafts|accepted_tokens" | head -40 > "$LOGD/${TAG}.metrics_excerpt.txt" || true
cat "$LOGD/${TAG}.metrics_excerpt.txt"
echo
log "=== AMX activation lines from boot log ==="
grep -E "CpuAmxProposer|cpu_amx|AMX|cpu_amx_draft" "$BOOT_LOG" | head -40
echo
log "=== DONE boot smoke for $TAG (engine kept running, port=$PORT) ==="
