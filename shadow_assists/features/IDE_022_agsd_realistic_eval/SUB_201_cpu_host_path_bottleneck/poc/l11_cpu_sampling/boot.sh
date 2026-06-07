#!/usr/bin/env bash
# L11 CPU sampling offload — Qwen2.5-7B (TP=1, GPU 7)
# mode = baseline | cpu_sampling
set -uo pipefail
MODE="${1:?usage: boot.sh baseline|cpu_sampling}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT="${PORT:-8011}"
GPUS="${GPUS:-7}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TP="${TP:-1}"

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l11_cpu_sampling
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

case "$MODE" in
  baseline)
    EXTRA_PREFIX="VLLM_CPU_SAMPLING=0"
    ;;
  cpu_sampling)
    EXTRA_PREFIX="VLLM_CPU_SAMPLING=1"
    ;;
  *) echo "unknown mode: $MODE"; exit 2;;
esac

BOOT_LOG="$LOGD/boot_${MODE}.log"
PID_FILE="$LOGD/${MODE}.pid"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "=== L11 boot $MODE port=$PORT gpus=$GPUS model=$MODEL ==="

ENV_PREFIX="CUDA_VISIBLE_DEVICES=$GPUS ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib:${LD_LIBRARY_PATH:-} $EXTRA_PREFIX"

CMD="$ENV_PREFIX setsid $VBIN serve $MODEL --tensor-parallel-size $TP --port $PORT --gpu-memory-utilization 0.85 --max-model-len 16384 --enforce-eager --no-async-scheduling"

log "CMD: $CMD"
bash -c "$CMD > '$BOOT_LOG' 2>&1 < /dev/null &
echo \$! > '$PID_FILE'"

PID=$(cat "$PID_FILE")
log "PID=$PID  log=$BOOT_LOG"

WAIT_READY_MAX="${WAIT_READY_MAX:-600}"
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
  if ! kill -0 "$PID" 2>/dev/null; then
    log "process died early"
    tail -80 "$BOOT_LOG"
    exit 3
  fi
  sleep 2
done
if [ "$READY" != "1" ]; then
  log "WAIT_READY timeout"
  tail -80 "$BOOT_LOG"
  exit 4
fi
log "boot OK PID=$PID PORT=$PORT MODE=$MODE"
