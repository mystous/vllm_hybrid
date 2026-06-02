#!/usr/bin/env bash
# TSK_042 oracle 측정 — 모델 × method(vanilla/suffix/ngram) phase 직렬 (TP=8).
# 각 phase: 백엔드 1개 기동 → oracle_runner(concurrency=1) → process-group kill → GPU free.
# 라우터 없음(단일 백엔드 격리 측정). run_full_8gpu.sh 의 헬퍼 패턴 재사용.
#
# 사용:
#   MODELS="Qwen/Qwen2.5-32B-Instruct" METHODS="vanilla suffix ngram" \
#   SAMPLED=/tmp/sampled_smoke.parquet LIMIT=20 OUTDIR=/tmp/oracle_smoke \
#   bash run_oracle_8gpu.sh
set -uo pipefail
cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval

MODELS="${MODELS:-Qwen/Qwen2.5-32B-Instruct}"
METHODS="${METHODS:-vanilla suffix ngram}"
SAMPLED="${SAMPLED:-/tmp/sampled_smoke.parquet}"
LIMIT="${LIMIT:-0}"                 # 0=전체 prompt
MAXTOK="${MAXTOK:-512}"
MML="${MML:-8192}"                  # 입력≤4096 + 출력512 → 8192 충분
OUTDIR="${OUTDIR:-/workspace/host_vllm_hybrid/$RE/runs/oracle_$(date +%Y%m%d_%H%M%S)}"
LOGD="$OUTDIR/_logs"; mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"; : > "$RAW"

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
wait_ready(){ for i in $(seq 1 240); do curl -sf "$1/v1/models" >/dev/null 2>&1 && { log "READY $1"; return 0; }; sleep 5; done; log "TIMEOUT $1"; return 1; }
kill_pgroup(){ local pid=$1; [ -z "$pid" ] && return 0; local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
wait_gpu_free(){ for i in $(seq 1 24); do local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${u:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }; sleep 5; done; return 0; }

spec_args(){ case "$1" in
  vanilla) echo "";;
  suffix)  echo "--speculative-config {\"method\":\"suffix\",\"num_speculative_tokens\":32}";;
  ngram)   echo "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":8,\"prompt_lookup_max\":4}";;
  *) echo "";; esac; }

log "=== oracle run → $OUTDIR (models=[$MODELS] methods=[$METHODS] limit=$LIMIT) ==="
for MODEL in $MODELS; do
  TAG=$(basename "$MODEL")
  for M in $METHODS; do
    log "### $TAG × $M ###"
    SA=$(spec_args "$M")
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid "$VBIN" serve "$MODEL" \
      --tensor-parallel-size 8 --port 8001 --gpu-memory-utilization 0.85 \
      --max-model-len "$MML" --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
      $SA > "$LOGD/${TAG}_${M}.log" 2>&1 < /dev/null &
    PID=$!
    if wait_ready http://127.0.0.1:8001; then
      PYTHONPATH=. "$PY" "$RE/oracle_runner.py" --in "$SAMPLED" --method "$M" \
        --model "$MODEL" --model-tag "$TAG" --port 8001 --max-tokens "$MAXTOK" \
        --limit "$LIMIT" --out "$RAW" >> "$LOGD/oracle.log" 2>&1
      log "     done $TAG × $M"
    else
      log "     SKIP $TAG × $M (backend timeout)"
    fi
    kill_pgroup "$PID"; wait_gpu_free
  done
done
log "=== building oracle_table ==="
PYTHONPATH=. "$PY" "$RE/build_oracle_table.py" --raw "$RAW" --out "$OUTDIR/oracle_table.parquet" | tee "$OUTDIR/SUMMARY.md"
log "=== DONE → $OUTDIR ==="
echo "$OUTDIR" > /tmp/oracle_run_dir.txt
