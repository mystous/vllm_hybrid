#!/usr/bin/env bash
# TSK_042 LHC Path 1 validation — generic per-model runner.
# Usage:
#   MODEL=<hf-id> TAG=<short> TP=<n> CONFIGS="vanilla lhc_path1" \
#   CORPORA="sharegpt swebench humaneval mbpp wildchat lmsys" \
#   SWEEPS="1 2 3" bash run_model.sh
#
# Mirrors tsk042_validation/run_validation.sh but with per-model out dirs.
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
SWEEPS="${SWEEPS:-1 2 3}"
LIMIT="${LIMIT:-500}"
CONC="${CONC:-32}"
MAXTOK="${MAXTOK:-8192}"
MML="${MML:-16384}"
PORT="${PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
OUTROOT="lhc_phase4/tsk042_10model_validation/${TAG}"
LOGD="$OUTROOT/_logs"; mkdir -p "$LOGD"

# Common envs (TSK_042 Optimal baseline replication).
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGD/run.log"; }

wait_ready(){
  local url=$1 pid=$2
  for i in $(seq 1 360); do
    curl -sf "$url/v1/models" >/dev/null 2>&1 && { log "READY $url"; return 0; }
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { log "DEAD backend (boot 실패) pid=$pid"; return 1; }
    sleep 5
  done
  log "TIMEOUT $url"; return 1
}

kill_pgroup(){
  # 우리가 setsid 로 spawn 한 PID 의 process group 만 정확히 정리한다.
  # 절대 broad pkill 사용 금지 — 병행 agent 의 vllm 까지 죽일 위험.
  local pid=$1
  [ -z "$pid" ] && return 0
  local pg
  pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$pg" ]; then
    # 같은 pgid 의 우리 descendant 만 kill
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
  log "GPU not fully freed but continuing"
  return 0
}

run_config(){
  local cfg=$1
  local outdir="$OUTROOT/${cfg}_runs"
  mkdir -p "$outdir"

  local extra_env=""
  if [ "$cfg" = "lhc_path1" ]; then
    extra_env="VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"
  fi

  log "### MODEL=$TAG CONFIG=$cfg TP=$TP (extra_env: ${extra_env:-none}) ###"

  log "    boot vllm serve $MODEL TP=$TP port=$PORT"
  local cuda_list
  if [ "$TP" = "4" ]; then cuda_list="0,1,2,3"; else cuda_list="0,1,2,3,4,5,6,7"; fi

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
    kill_pgroup "$PID"; wait_gpu_free
    return 1
  fi

  for SW in $SWEEPS; do
    for C in $CORPORA; do
      local outjson="$outdir/${C}_s${SW}.json"
      local rawjson="$outdir/${C}_s${SW}.raw.jsonl"
      log "    [$cfg s$SW] corpus=$C"
      PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method vanilla \
        --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$MAXTOK" \
        --concurrency "$CONC" --corpus "$C" \
        --out "$outjson" --raw "$rawjson" \
        >> "$LOGD/${cfg}_bench.log" 2>&1 || log "    FAILED corpus=$C s=$SW"
    done
    # mix (shuffle limit=500)
    local outjson="$outdir/mix_s${SW}.json"
    local rawjson="$outdir/mix_s${SW}.raw.jsonl"
    log "    [$cfg s$SW] corpus=mix"
    PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method vanilla \
      --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$MAXTOK" \
      --concurrency "$CONC" --limit "$LIMIT" --shuffle --seed "$((42 + SW))" \
      --out "$outjson" --raw "$rawjson" \
      >> "$LOGD/${cfg}_bench.log" 2>&1 || log "    FAILED corpus=mix s=$SW"
  done

  log "    teardown $cfg (PID=$PID)"
  kill_pgroup "$PID"; wait_gpu_free
  return 0
}

log "=== TSK_042 10-model validation start (model=$MODEL tag=$TAG TP=$TP configs=[$CONFIGS] corpora=[$CORPORA] sweeps=[$SWEEPS]) ==="
for CFG in $CONFIGS; do
  run_config "$CFG"
done
log "=== model=$TAG DONE → $OUTROOT ==="
