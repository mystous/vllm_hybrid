#!/usr/bin/env bash
# TSK_042 unified per-model runner.
# Inputs (env): MODEL, TAG, TP, SWEEPS (space-separated), CORPORA (space-separated; mix appended unless CORPORA_EXACT=1)
# CONFIGS="vanilla lhc_path1" (default both). GPU_MEM_UTIL=0.92 default.
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
SAMPLED=vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

MODEL="${MODEL:?MODEL required}"
TAG="${TAG:-$(basename "$MODEL")}"
TP="${TP:-8}"
CONFIGS="${CONFIGS:-vanilla lhc_path1}"
CORPORA="${CORPORA:-sharegpt swebench humaneval mbpp wildchat lmsys}"
CORPORA_EXACT="${CORPORA_EXACT:-0}"  # if 1, skip the mix corpus
SWEEPS="${SWEEPS:-1 2 3}"
LIMIT="${LIMIT:-500}"
CONC="${CONC:-32}"
MAXTOK="${MAXTOK:-8192}"
MML="${MML:-16384}"
PORT="${PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
OUTROOT="lhc_phase4/tsk042_10model_unified/${TAG}"
LOGD="$OUTROOT/_logs"; mkdir -p "$LOGD"

# Optimal baseline envs (TSK_042).
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGD/run.log"; }

wait_ready(){
  local url=$1 pid=$2
  for i in $(seq 1 720); do  # 60 min boot timeout for 405B / 671B
    curl -sf "$url/v1/models" >/dev/null 2>&1 && { log "READY $url"; return 0; }
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { log "DEAD backend (boot fail) pid=$pid"; return 1; }
    sleep 5
  done
  log "TIMEOUT $url"; return 1
}

kill_pgroup(){
  local pid=$1
  [ -z "$pid" ] && return 0
  local pg
  pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$pg" ]; then
    kill -9 -"$pg" 2>/dev/null || true
  fi
  kill -9 "$pid" 2>/dev/null || true
  sleep 3
}

wait_gpu_free(){
  for i in $(seq 1 60); do
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    [ "${u:-1}" -lt 8000 ] && { log "GPU freed (used=${u} MiB total)"; return 0; }
    sleep 5
  done
  log "GPU not fully freed (continuing)"
  return 0
}

kill_orphans(){
  # Orphan vllm/python compute apps still holding GPU mem.
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u)
  for p in $pids; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
  done
  sleep 3
}

run_config(){
  local cfg=$1
  local outdir="$OUTROOT/${cfg}_runs"
  mkdir -p "$outdir"

  # Check if config already complete (all expected output files exist).
  local need_run=0
  for SW in $SWEEPS; do
    for C in $CORPORA; do
      [ -f "$outdir/${C}_s${SW}.json" ] || need_run=1
    done
    if [ "$CORPORA_EXACT" != "1" ]; then
      [ -f "$outdir/mix_s${SW}.json" ] || need_run=1
    fi
  done
  if [ "$need_run" = "0" ]; then
    log "    [$cfg] all output files present — skipping boot"
    return 0
  fi

  local extra_env=""
  if [ "$cfg" = "lhc_path1" ]; then
    extra_env="VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"
  fi

  log "### MODEL=$TAG CONFIG=$cfg TP=$TP (extra_env: ${extra_env:-none}) ###"

  log "    boot vllm serve $MODEL TP=$TP port=$PORT"
  local cuda_list
  case "$TP" in
    1) cuda_list="0";;
    2) cuda_list="0,1";;
    4) cuda_list="0,1,2,3";;
    *) cuda_list="0,1,2,3,4,5,6,7";;
  esac

  if [ -n "$extra_env" ]; then
    CUDA_VISIBLE_DEVICES="$cuda_list" \
    env $extra_env \
    setsid "$VBIN" serve "$MODEL" \
      --tensor-parallel-size "$TP" --port "$PORT" --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MML" --max-num-seqs "$CONC" --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
      > "$LOGD/${cfg}_serve.log" 2>&1 < /dev/null &
  else
    CUDA_VISIBLE_DEVICES="$cuda_list" \
    setsid "$VBIN" serve "$MODEL" \
      --tensor-parallel-size "$TP" --port "$PORT" --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MML" --max-num-seqs "$CONC" --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
      > "$LOGD/${cfg}_serve.log" 2>&1 < /dev/null &
  fi
  local PID=$!
  log "    serve PID=$PID"

  if ! wait_ready "http://127.0.0.1:$PORT" "$PID"; then
    log "    SKIP $cfg (boot fail)"
    kill_pgroup "$PID"; kill_orphans; wait_gpu_free
    return 1
  fi

  for SW in $SWEEPS; do
    for C in $CORPORA; do
      local outjson="$outdir/${C}_s${SW}.json"
      local rawjson="$outdir/${C}_s${SW}.raw.jsonl"
      if [ -f "$outjson" ]; then
        log "    [$cfg s$SW] corpus=$C (cached, skipping)"
        continue
      fi
      log "    [$cfg s$SW] corpus=$C"
      PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method vanilla \
        --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$MAXTOK" \
        --concurrency "$CONC" --corpus "$C" \
        --out "$outjson" --raw "$rawjson" \
        >> "$LOGD/${cfg}_bench.log" 2>&1 || log "    FAILED corpus=$C s=$SW"
    done
    if [ "$CORPORA_EXACT" != "1" ]; then
      local outjson="$outdir/mix_s${SW}.json"
      local rawjson="$outdir/mix_s${SW}.raw.jsonl"
      if [ -f "$outjson" ]; then
        log "    [$cfg s$SW] corpus=mix (cached, skipping)"
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
  kill_pgroup "$PID"; kill_orphans; wait_gpu_free
  return 0
}

log "=== TSK_042 unified run model=$MODEL tag=$TAG TP=$TP configs=[$CONFIGS] corpora=[$CORPORA] sweeps=[$SWEEPS] ==="
for CFG in $CONFIGS; do
  run_config "$CFG"
done
touch "$OUTROOT/.done"
log "=== model=$TAG DONE → $OUTROOT ==="
