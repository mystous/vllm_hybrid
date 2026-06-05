#!/usr/bin/env bash
# TSK_042 (B) canonical 처리량 — 모델 × method phase 직렬 (TP=8, conc=32, max_tokens=8192).
# 실 trace 자연 입력. per (model,method) summary json + 합본 raw → 집계 표.
set -uo pipefail
cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval

MODELS="${MODELS:?MODELS 필요}"
METHODS="${METHODS:-vanilla suffix ngram}"
CORPORA="${CORPORA:-sharegpt swebench humaneval mbpp wildchat lmsys}"
SAMPLED="${SAMPLED:?SAMPLED 필요}"
LIMIT="${LIMIT:-500}"
CONC="${CONC:-32}"
MAXTOK="${MAXTOK:-8192}"
MML="${MML:-16384}"          # 실 입력(≤4096) + 8192 출력
OUTDIR="${OUTDIR:?OUTDIR 필요}"
LOGD="$OUTDIR/_logs"; mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"; : > "$RAW"

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
wait_ready(){ for i in $(seq 1 180); do curl -sf "$1/v1/models" >/dev/null 2>&1 && { log "READY $1"; return 0; }; [ -n "${2:-}" ] && ! kill -0 "$2" 2>/dev/null && { log "DEAD backend (boot 실패) $1"; return 1; }; sleep 5; done; log "TIMEOUT $1"; return 1; }
kill_pgroup(){ local pid=$1; [ -z "$pid" ] && return 0; local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
wait_gpu_free(){ for i in $(seq 1 24); do local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${u:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }; sleep 5; done; return 0; }
spec_args(){ case "$1" in
  vanilla) echo "";;
  suffix)  echo "--speculative-config {\"method\":\"suffix\",\"num_speculative_tokens\":32}";;
  ngram)   echo "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":8,\"prompt_lookup_max\":4}";;
  *) echo "";; esac; }
pick_tp(){ # head 수 %8==0 → TP8, 아니면 TP4
  local h; h=$("$PY" -c "import glob,json,os,sys;c=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--'+sys.argv[1].replace('/','--')+'/snapshots/*/config.json'));print(json.load(open(c[0])).get('num_attention_heads',0) if c else 0)" "$1" 2>/dev/null)
  if [ "${h:-0}" -gt 0 ] && [ $((h % 8)) -eq 0 ]; then echo 8; else echo 4; fi; }

log "=== throughput run → $OUTDIR (models=[$MODELS] methods=[$METHODS] limit=$LIMIT conc=$CONC max=$MAXTOK) ==="
for MODEL in $MODELS; do
  TAG=$(basename "$MODEL")
  for M in $METHODS; do
    TP=$(pick_tp "$MODEL"); [ "$TP" = 8 ] && GPUS=0,1,2,3,4,5,6,7 || GPUS=0,1,2,3
    log "### $TAG × $M (TP=$TP) ###"
    SA=$(spec_args "$M")
    CUDA_VISIBLE_DEVICES=$GPUS setsid "$VBIN" serve "$MODEL" \
      --tensor-parallel-size $TP --port 8001 --gpu-memory-utilization 0.85 \
      --max-model-len "$MML" --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
      $SA > "$LOGD/${TAG}_${M}.log" 2>&1 < /dev/null &
    PID=$!
    if wait_ready http://127.0.0.1:8001 "$PID"; then
      # 6 corpus 격리
      for C in $CORPORA; do
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$M" \
          --model "$MODEL" --model-tag "$TAG" --port 8001 --max-tokens "$MAXTOK" \
          --concurrency "$CONC" --corpus "$C" \
          --out "$OUTDIR/summ_${TAG}_${M}_${C}.json" --raw "$RAW" >> "$LOGD/tput.log" 2>&1
      done
      # mix (shuffle 500)
      PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$M" \
        --model "$MODEL" --model-tag "$TAG" --port 8001 --max-tokens "$MAXTOK" \
        --concurrency "$CONC" --limit "$LIMIT" --shuffle \
        --out "$OUTDIR/summ_${TAG}_${M}_mix.json" --raw "$RAW" >> "$LOGD/tput.log" 2>&1
      log "     done $TAG × $M (corpus×6 + mix)"
    else
      log "     SKIP $TAG × $M (backend boot 실패)"
    fi
    kill_pgroup "$PID"; wait_gpu_free
  done
done
log "=== aggregate ==="
PYTHONPATH=. "$PY" "$RE/build_throughput_table.py" --dir "$OUTDIR" | tee "$OUTDIR/SUMMARY.md"
log "=== DONE → $OUTDIR ==="
echo "$OUTDIR" > /tmp/tput_run_dir.txt
