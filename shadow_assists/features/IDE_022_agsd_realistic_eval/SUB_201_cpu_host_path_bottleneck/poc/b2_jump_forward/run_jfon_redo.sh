#!/usr/bin/env bash
# Re-run JF-on bench + correctness on a single engine boot (faster) with diag logs.
set -uo pipefail
cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8007
GPUS="6,7"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet
POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b2_jump_forward
LOGD="$POC_DIR/_logs"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1==6||$1==7 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
[ "${USED:-0}" -gt 4000 ] && { log "ABORT busy=$USED"; exit 1; }

log "boot JF=1"
CUDA_VISIBLE_DEVICES=$GPUS VLLM_USE_XGRAMMAR_JUMP_FORWARD=1 \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 2 --port "$PORT" \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --allow-deprecated-quantization \
    > "$LOGD/redo_boot_jfon.log" 2>&1 < /dev/null &
PID=$!
echo $PID > "$LOGD/redo_pid"
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "READY in ${i}s"; break; }
  sleep 1
done

log "bench constrained_jf"
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" "$POC_DIR/constrained_runner.py" \
    --in "$PARQ" --mode json_schema --model "$MODEL" \
    --model-tag "Llama-3.1-8B-Instruct" --port "$PORT" \
    --max-tokens 512 --concurrency 16 --limit 200 --corpus sharegpt \
    --out "$POC_DIR/llama8b_constrained_jf.json" \
    --raw "$POC_DIR/llama8b_constrained_jf.raw.jsonl" \
  2>&1 | tee "$LOGD/redo_bench_jf.log"

log "correctness JF on"
"$PY" "$POC_DIR/correctness_check.py" \
  --in "$PARQ" --port "$PORT" --model "$MODEL" --limit 10 \
  --label on --out "$POC_DIR/correctness_on.json" \
  2>&1 | tee "$LOGD/redo_correctness_on.log" || true

# Capture JF telemetry + diag
grep -E "jump-forward|JF" "$LOGD/redo_boot_jfon.log" > "$LOGD/redo_jf_telemetry.txt" 2>/dev/null || true
log "JF lines captured:"
wc -l "$LOGD/redo_jf_telemetry.txt"
tail -20 "$LOGD/redo_jf_telemetry.txt"

log "kill engine pid=$PID"
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
[ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null
kill -9 "$PID" 2>/dev/null
sleep 3
for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
  cmd=$(cat /proc/$op/cmdline 2>/dev/null | tr '\0' ' ')
  if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then kill -9 "$op" 2>/dev/null; fi
done
for i in $(seq 1 20); do
  u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1==6||$1==7 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  [ "${u:-1}" -lt 1000 ] && { log "GPU freed"; break; }
  sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1==6||$1==7' | tee "$LOGD/redo_gpu_after.txt"
log "DONE"
