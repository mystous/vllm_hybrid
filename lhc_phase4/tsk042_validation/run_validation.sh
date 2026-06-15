#!/usr/bin/env bash
# TSK_042 LHC Path 1 real-trace validation.
# 7 corpus × {vanilla, lhc_path1} × 3 sweeps = 42 bench
# model: Llama-3.1-8B-Instruct, TP=8, conc=32, max_tokens=8192.
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
SAMPLED=vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TAG=$(basename "$MODEL")
CONFIGS="${CONFIGS:-vanilla lhc_path1}"
CORPORA="${CORPORA:-sharegpt swebench humaneval mbpp wildchat lmsys}"
SWEEPS="${SWEEPS:-1 2 3}"
LIMIT="${LIMIT:-500}"
CONC="${CONC:-32}"
MAXTOK="${MAXTOK:-8192}"
MML="${MML:-16384}"
PORT="${PORT:-8001}"
OUTROOT=lhc_phase4/tsk042_validation
LOGD="$OUTROOT/_logs"; mkdir -p "$LOGD"

# Common envs (TSK_042 Optimal baseline replication).
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGD/run.log"; }

wait_ready(){
  local url=$1 pid=$2
  for i in $(seq 1 180); do
    curl -sf "$url/v1/models" >/dev/null 2>&1 && { log "READY $url"; return 0; }
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { log "DEAD backend (boot 실패) pid=$pid"; return 1; }
    sleep 5
  done
  log "TIMEOUT $url"; return 1
}

kill_pgroup(){
  local pid=$1
  [ -z "$pid" ] && return 0
  local pg
  pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
  kill -9 "$pid" 2>/dev/null
  # 잔존 vllm 자식 정리
  pkill -9 -f "vllm.entrypoints" 2>/dev/null
  pkill -9 -f "VLLM::" 2>/dev/null
  sleep 2
}

wait_gpu_free(){
  for i in $(seq 1 30); do
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    [ "${u:-1}" -lt 4000 ] && { log "GPU freed (used=${u} MiB)"; return 0; }
    sleep 5
  done
  log "GPU not fully freed but continuing"
  return 0
}

run_config(){
  local cfg=$1
  local outdir="$OUTROOT/${cfg}_runs"
  mkdir -p "$outdir"

  # Per-config env layer (subshell exports applied via inline assignments below)
  local extra_env=""
  if [ "$cfg" = "lhc_path1" ]; then
    extra_env="VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"
  fi

  log "### CONFIG=$cfg (extra_env: ${extra_env:-none}) ###"

  # vllm serve (TP=8)
  log "    boot vllm serve $MODEL TP=8 port=$PORT"
  if [ -n "$extra_env" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    env $extra_env \
    setsid "$VBIN" serve "$MODEL" \
      --tensor-parallel-size 8 --port "$PORT" --gpu-memory-utilization 0.92 \
      --max-model-len "$MML" --max-num-seqs "$CONC" --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
      > "$LOGD/${cfg}_serve.log" 2>&1 < /dev/null &
  else
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    setsid "$VBIN" serve "$MODEL" \
      --tensor-parallel-size 8 --port "$PORT" --gpu-memory-utilization 0.92 \
      --max-model-len "$MML" --max-num-seqs "$CONC" --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
      > "$LOGD/${cfg}_serve.log" 2>&1 < /dev/null &
  fi
  local PID=$!
  log "    serve PID=$PID"

  if ! wait_ready "http://127.0.0.1:$PORT" "$PID"; then
    log "    SKIP $cfg (boot fail)"
    kill_pgroup "$PID"; wait_gpu_free
    return 1
  fi

  # Run each corpus × sweep
  for SW in $SWEEPS; do
    for C in $CORPORA; do
      local outjson="$outdir/${C}_s${SW}.json"
      local rawjson="$outdir/${C}_s${SW}.raw.jsonl"
      if [ -s "$outjson" ] && [ "${SKIP_EXISTING:-1}" = "1" ]; then
        log "    [$cfg s$SW] corpus=$C SKIP (exists)"
        continue
      fi
      log "    [$cfg s$SW] corpus=$C"
      PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method vanilla \
        --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$MAXTOK" \
        --concurrency "$CONC" --corpus "$C" \
        --out "$outjson" --raw "$rawjson" \
        >> "$LOGD/${cfg}_bench.log" 2>&1 || log "    FAILED corpus=$C s=$SW"
    done
    # mix (shuffle limit=500)
    if [ "${RUN_MIX:-1}" = "1" ]; then
      local outjson="$outdir/mix_s${SW}.json"
      local rawjson="$outdir/mix_s${SW}.raw.jsonl"
      if [ -s "$outjson" ] && [ "${SKIP_EXISTING:-1}" = "1" ]; then
        log "    [$cfg s$SW] corpus=mix SKIP (exists)"
      else
        log "    [$cfg s$SW] corpus=mix"
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method vanilla \
          --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$MAXTOK" \
          --concurrency "$CONC" --limit "$LIMIT" --shuffle --seed "$((42 + SW))" \
          --out "$outjson" --raw "$rawjson" \
          >> "$LOGD/${cfg}_bench.log" 2>&1 || log "    FAILED corpus=mix s=$SW"
      fi
    fi
  done

  log "    teardown $cfg (PID=$PID)"
  kill_pgroup "$PID"; wait_gpu_free
  return 0
}

log "=== TSK_042 validation start (model=$MODEL configs=[$CONFIGS] corpora=[$CORPORA] sweeps=[$SWEEPS]) ==="
for CFG in $CONFIGS; do
  run_config "$CFG"
done
log "=== TSK_042 validation DONE → $OUTROOT ==="
