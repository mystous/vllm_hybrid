#!/usr/bin/env bash
# SUB_213 multi-model pad sweep — 전 모델 × K{4,6,8,12} pad × 7 corpus
# 목적: 각 모델별 최적 padding K 를 구해 SUB_212 6-point 표에 ⑦ 열로 추가.
# 70B 는 runs_fullmatrix 에서 완료 → 제외. 소→대 순서 (조기 결과 + 큰 모델은 마지막).
set -u
cd /home/mystous/vllm_hybrid
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
SUBD=shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform
OUTDIR=$SUBD/runs_multimodel; LOGD="$OUTDIR/_logs"; mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"; : > "$RAW"
SAMPLED="$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet"
CORPORA=(sharegpt swebench humaneval mbpp wildchat lmsys)
PORT=8011 MML=16384 CONC=32 MAXTOK=8192 LIMIT=500
VLLM_CPUS="0-47,56-103"
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1
log(){ echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

# tag|model_id|TP  (SUB_212 §7, 70B 제외)
MODELS=(
  "Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct|4"
  "DeepSeek-R1-Distill-Qwen-7B|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|4"
  "Llama-3.1-8B-Instruct|meta-llama/Llama-3.1-8B-Instruct|8"
  "Qwen2.5-32B-Instruct|Qwen/Qwen2.5-32B-Instruct|8"
  "DeepSeek-R1-Distill-Qwen-32B|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|8"
  "Qwen2.5-72B-Instruct|Qwen/Qwen2.5-72B-Instruct|8"
  "DeepSeek-R1-Distill-Llama-70B|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|8"
  "Llama-3.1-405B-Instruct-FP8|meta-llama/Llama-3.1-405B-Instruct-FP8|8"
  "DeepSeek-R1|deepseek-ai/DeepSeek-R1|8"
)

wait_ready(){ local i; for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log READY; return 0; }
    kill -0 "$1" 2>/dev/null || { log "DEAD backend"; return 1; }; sleep 5; done; log TIMEOUT; return 1; }
kill_pg(){ local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
wait_gpu(){ local i u; for i in $(seq 1 80); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}')
    [ "${u:-1}" -lt 4000 ] && return 0; sleep 5; done; return 0; }
trap 'echo "[trap] interrupted"; kill_pg ${PID:-}; exit 130' INT TERM

log "=== multi-model pad sweep start: ${#MODELS[@]} models x K{4,6,8,12}pad x 7corpus ==="
for ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r TAG MODEL TP <<< "$ENTRY"
    GPUS=$(seq -s, 0 $((TP-1)))
    for K in 4 6 8 12; do
        CELL="${TAG}_k${K}pad"
        # 7 corpus 모두 존재하면 skip
        DONE=$(ls "$OUTDIR"/summ_${TAG}_k${K}pad_*.json 2>/dev/null | wc -l)
        [ "$DONE" -ge 7 ] && { log "skip $CELL (done)"; continue; }
        BL="$LOGD/boot_${CELL}.log"; : > "$BL"
        env CUDA_VISIBLE_DEVICES=$GPUS VLLM_SUFFIX_PAD_UNIFORM=1 setsid taskset -c "$VLLM_CPUS" \
            "$VBIN" serve "$MODEL" --tensor-parallel-size $TP --port $PORT \
            --gpu-memory-utilization 0.85 --max-model-len $MML \
            --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
            --speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":$K}" \
            > "$BL" 2>&1 < /dev/null &
        PID=$!
        log "boot $CELL (TP=$TP gpus=$GPUS pid=$PID)"
        if wait_ready "$PID"; then
            for C in "${CORPORA[@]}"; do
                OUT="$OUTDIR/summ_${TAG}_k${K}pad_${C}.json"
                [ -s "$OUT" ] && continue
                PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "k${K}pad" \
                    --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
                    --concurrency $CONC --corpus "$C" --out "$OUT" --raw "$RAW" \
                    >> "$LOGD/bench_${CELL}.log" 2>&1 || log "  bench FAIL $CELL x $C"
            done
            OUT="$OUTDIR/summ_${TAG}_k${K}pad_mix.json"
            [ -s "$OUT" ] || PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "k${K}pad" \
                --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
                --concurrency $CONC --limit $LIMIT --shuffle --out "$OUT" --raw "$RAW" \
                >> "$LOGD/bench_${CELL}.log" 2>&1 || log "  bench FAIL $CELL x mix"
            log "  done $CELL"
        else
            log "  SKIP $CELL (boot fail)"; tail -25 "$BL"
        fi
        kill_pg "$PID"; wait_gpu
    done
done
log "=== multi-model pad sweep done ==="
