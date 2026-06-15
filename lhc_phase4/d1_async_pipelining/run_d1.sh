#!/usr/bin/env bash
# D-1 async pipelining lever 측정 — Llama-3.1-70B TP=8.
# 결론적으로 vLLM 의 baseline 이 이미 async_scheduling=True + batch_queue_size=2
# (step_with_batch_queue) 로 D-1 의 본질 (prev step host op vs next step GPU
# execute overlap) 을 native 로 active.
# 본 script 는 추가로 가능한 lever 들을 측정 — 모두 server-side flag 만 변경
# (코드 수정 없음).
#
# LEVERS:
#   L1: --stream-interval 16   (host IPC 빈도 1/16 로 축소; SSE event coalesce)
#   L2: (선택) --max-num-batched-tokens 증가 — 별도 측정 가능
#
# 측정 조건:
#   - workload: TSK_042 sampled_prompts.parquet conc=32 max_tok=8192 stream
#   - corpus: 7 (sharegpt swebench humaneval mbpp wildchat lmsys + mix)
#   - sweep: s1 only (빠른 양수 진단 — 양수면 s2/s3 확장)
#   - lever 비교: baseline (lhc_phase4/tsk042_10model_unified/Llama-3.1-70B-Instruct/vanilla_runs/) 와 paired
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
SAMPLED=vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

MODEL="meta-llama/Llama-3.1-70B-Instruct"
TAG="Llama-3.1-70B-Instruct"
TP=8
CONC=32
MAXTOK=8192
MML=16384
PORT=8011
GPU_MEM_UTIL=0.92
CORPORA="sharegpt swebench humaneval mbpp wildchat lmsys"
SWEEPS="1"
LIMIT=500

LEVER="${LEVER:-L1_stream16}"
STREAM_INTERVAL="${STREAM_INTERVAL:-16}"

OUTROOT="lhc_phase4/d1_async_pipelining"
LOGD="$OUTROOT/logs/${LEVER}"
OUTD="$OUTROOT/runs/${LEVER}"
mkdir -p "$LOGD" "$OUTD"

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGD/run.log"; }

wait_ready(){
  local url=$1 pid=$2
  for i in $(seq 1 720); do
    curl -sf "$url/v1/models" >/dev/null 2>&1 && { log "READY $url"; return 0; }
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { log "DEAD backend pid=$pid"; return 1; }
    sleep 5
  done
  log "TIMEOUT $url"; return 1
}

kill_pgroup(){
  local pid=$1
  [ -z "$pid" ] && return 0
  local pg
  pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null || true
  kill -9 "$pid" 2>/dev/null || true
  sleep 3
}

kill_orphans(){
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u)
  for p in $pids; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
  done
  sleep 3
}

log "=== D-1 lever=$LEVER stream_interval=$STREAM_INTERVAL model=$TAG TP=$TP ==="
log "    boot vllm serve ..."

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
setsid "$VBIN" serve "$MODEL" \
  --tensor-parallel-size "$TP" --port "$PORT" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MML" --max-num-seqs "$CONC" --enable-prefix-caching \
  --stream-interval "$STREAM_INTERVAL" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  > "$LOGD/serve.log" 2>&1 < /dev/null &
PID=$!
log "    serve PID=$PID"

if ! wait_ready "http://127.0.0.1:$PORT" "$PID"; then
  log "    boot fail, abort."
  kill_pgroup "$PID"; kill_orphans
  exit 1
fi

for SW in $SWEEPS; do
  for C in $CORPORA; do
    outjson="$OUTD/${C}_s${SW}.json"
    rawjson="$OUTD/${C}_s${SW}.raw.jsonl"
    if [ -f "$outjson" ]; then
      log "    [s$SW] $C cached, skipping"
      continue
    fi
    log "    [s$SW] corpus=$C"
    PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$LEVER" \
      --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$MAXTOK" \
      --concurrency "$CONC" --corpus "$C" \
      --out "$outjson" --raw "$rawjson" \
      >> "$LOGD/bench.log" 2>&1 || log "    FAILED corpus=$C s=$SW"
  done
  outjson="$OUTD/mix_s${SW}.json"
  rawjson="$OUTD/mix_s${SW}.raw.jsonl"
  if [ -f "$outjson" ]; then
    log "    [s$SW] mix cached, skipping"
  else
    log "    [s$SW] corpus=mix"
    PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$LEVER" \
      --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$MAXTOK" \
      --concurrency "$CONC" --limit "$LIMIT" --shuffle --seed "$((42 + SW))" \
      --out "$outjson" --raw "$rawjson" \
      >> "$LOGD/bench.log" 2>&1 || log "    FAILED corpus=mix s=$SW"
  fi
done

log "    teardown (PID=$PID)"
kill_pgroup "$PID"; kill_orphans
log "=== D-1 lever=$LEVER DONE → $OUTD ==="
