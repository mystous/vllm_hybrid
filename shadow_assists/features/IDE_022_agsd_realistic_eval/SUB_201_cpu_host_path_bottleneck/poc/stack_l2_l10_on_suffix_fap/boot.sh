#!/usr/bin/env bash
# Stack L2 + L10 on suffix + FaP — boot one configuration on B200 GPU 0,1 (TP=2)
#
# usage:
#   boot.sh A   # PIECEWISE + vanilla   (baseline)
#   boot.sh B   # PIECEWISE + suffix K=7
#   boot.sh C   # FaP       + suffix K=7
#   boot.sh D   # FaP       + suffix K=7 + L2 (VLLM_PREFETCH_TOKENIZE=1)
#   boot.sh E   # FaP       + suffix K=7 + L2 + L10
set -uo pipefail
RUN="${1:?usage: boot.sh A|B|C|D|E}"

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/stack_l2_l10_on_suffix_fap
LOGD="$ROOT/_logs"
RUNS="$ROOT/runs"
mkdir -p "$LOGD" "$RUNS"

VLLM=/workspace/vllm_dev_prj/bin/vllm
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8009
TP=2
GPUS="0,1"
MAX_MODEL_LEN=16384

CGMODE="FULL_AND_PIECEWISE"
SPEC=""
L2=0
L10=0
case "$RUN" in
  A) CGMODE="PIECEWISE";          SPEC="";                                                  L2=0; L10=0 ;;
  B) CGMODE="PIECEWISE";          SPEC='{"method":"suffix","num_speculative_tokens":7}';    L2=0; L10=0 ;;
  C) CGMODE="FULL_AND_PIECEWISE"; SPEC='{"method":"suffix","num_speculative_tokens":7}';    L2=0; L10=0 ;;
  D) CGMODE="FULL_AND_PIECEWISE"; SPEC='{"method":"suffix","num_speculative_tokens":7}';    L2=1; L10=0 ;;
  E) CGMODE="FULL_AND_PIECEWISE"; SPEC='{"method":"suffix","num_speculative_tokens":7}';    L2=1; L10=1 ;;
  *) echo "unknown RUN=$RUN"; exit 2 ;;
esac

BOOT_LOG="$LOGD/boot_${RUN}.log"
PID_FILE="$LOGD/${RUN}.pid"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# build env exports for the subshell
export CUDA_VISIBLE_DEVICES="$GPUS"
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
if [ "$L2" = "1" ]; then
  export VLLM_PREFETCH_TOKENIZE=1
  export VLLM_PREFETCH_TOKENIZE_WORKERS=2
else
  unset VLLM_PREFETCH_TOKENIZE VLLM_PREFETCH_TOKENIZE_WORKERS
fi
if [ "$L10" = "1" ]; then
  export VLLM_BURST_AWARE_ADMISSION=1
else
  unset VLLM_BURST_AWARE_ADMISSION
fi

CGCFG="{\"cudagraph_mode\":\"$CGMODE\"}"

log "=== STACK boot RUN=$RUN cgmode=$CGMODE spec='$SPEC' L2=$L2 L10=$L10 port=$PORT gpus=$GPUS tp=$TP ==="
log "boot log: $BOOT_LOG"
log "compilation-config: $CGCFG"
log "speculative-config: $SPEC"

# launch as array; setsid runs the process in its own session so kill_pgroup works
ARGS=( serve "$MODEL"
  --tensor-parallel-size "$TP"
  --port "$PORT"
  --gpu-memory-utilization 0.85
  --max-model-len "$MAX_MODEL_LEN"
  --allow-deprecated-quantization
  --compilation-config "$CGCFG"
)
if [ -n "$SPEC" ]; then
  ARGS+=( --speculative-config "$SPEC" )
fi

setsid "$VLLM" "${ARGS[@]}" > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
log "PID=$PID"

WAIT_READY_MAX="${WAIT_READY_MAX:-900}"
T_START=$(date +%s)
READY=0
for i in $(seq 1 $((WAIT_READY_MAX/2))); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    BOOT_SEC=$(( $(date +%s) - T_START ))
    log "READY in ${BOOT_SEC}s"
    echo "$BOOT_SEC" > "$LOGD/${RUN}.boot_sec"
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    log "process died early"
    tail -120 "$BOOT_LOG"
    exit 3
  fi
  sleep 2
done
if [ "$READY" != "1" ]; then
  log "WAIT_READY timeout after ${WAIT_READY_MAX}s"
  tail -120 "$BOOT_LOG"
  exit 4
fi
log "boot OK RUN=$RUN PID=$PID"
