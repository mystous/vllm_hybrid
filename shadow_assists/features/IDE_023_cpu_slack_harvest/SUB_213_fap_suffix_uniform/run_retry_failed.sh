#!/usr/bin/env bash
# SUB_213 재시도 — 실패 셀만 설정 변경해 재실행
#   원인: (a) 671B = 부팅 15~25분(DeepGEMM warmup 1827) > 러너 readiness 25분 한계 → DEAD 오판
#         (b) 405B-FP8 = fbgemm_fp8 deprecated → --allow-deprecated-quantization 필요
#   수정: readiness 50분(600×5s) + taskset 해제(전 코어, warmup 가속) + 405B 플래그
set -u
cd /home/mystous/vllm_hybrid
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
SUBD=shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform
OUTDIR=$SUBD/runs_multimodel; LOGD="$OUTDIR/_logs"; mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"
SAMPLED="$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet"
CORPORA=(sharegpt swebench humaneval mbpp wildchat lmsys)
PORT=8011 MML=16384 CONC=32 MAXTOK=8192 LIMIT=500
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1
log(){ echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

# tag|model_id|TP|extra_serve_args|Ks
JOBS=(
  "Llama-3.1-405B-Instruct-FP8|meta-llama/Llama-3.1-405B-Instruct-FP8|8|--allow-deprecated-quantization|4 6 8 12"
  "DeepSeek-R1|deepseek-ai/DeepSeek-R1|8||4 6"
)

wait_ready(){ local i; for i in $(seq 1 600); do   # 50분
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "READY ($((i*5))s)"; return 0; }
    kill -0 "$1" 2>/dev/null || { log "DEAD backend"; return 1; }; sleep 5; done; log TIMEOUT; return 1; }
kill_pg(){ local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
wait_gpu(){ local i u; for i in $(seq 1 100); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}')
    [ "${u:-1}" -lt 4000 ] && return 0; sleep 5; done; return 0; }
trap 'echo "[trap] interrupted"; kill_pg ${PID:-}; exit 130' INT TERM

log "=== retry failed cells start (readiness 50m, taskset 해제) ==="
for ENTRY in "${JOBS[@]}"; do
    IFS='|' read -r TAG MODEL TP EXTRA KS <<< "$ENTRY"
    GPUS=$(seq -s, 0 $((TP-1)))
    for K in $KS; do
        CELL="${TAG}_k${K}pad"
        DONE=$(ls "$OUTDIR"/summ_${TAG}_k${K}pad_*.json 2>/dev/null | wc -l)
        [ "$DONE" -ge 7 ] && { log "skip $CELL (done)"; continue; }
        BL="$LOGD/retry_boot_${CELL}.log"; : > "$BL"
        # taskset 없음 (전 코어), --allow-deprecated-quantization 등 EXTRA
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES=$GPUS VLLM_SUFFIX_PAD_UNIFORM=1 setsid \
            "$VBIN" serve "$MODEL" --tensor-parallel-size $TP --port $PORT \
            --gpu-memory-utilization 0.85 --max-model-len $MML \
            --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
            --speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":$K}" \
            $EXTRA > "$BL" 2>&1 < /dev/null &
        PID=$!
        log "boot $CELL (TP=$TP extra='$EXTRA' pid=$PID)"
        if wait_ready "$PID"; then
            for C in "${CORPORA[@]}"; do
                OUT="$OUTDIR/summ_${TAG}_k${K}pad_${C}.json"
                [ -s "$OUT" ] && continue
                PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "k${K}pad" \
                    --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
                    --concurrency $CONC --corpus "$C" --out "$OUT" --raw "$RAW" \
                    >> "$LOGD/retry_bench_${CELL}.log" 2>&1 || log "  bench FAIL $CELL x $C"
            done
            OUT="$OUTDIR/summ_${TAG}_k${K}pad_mix.json"
            [ -s "$OUT" ] || PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "k${K}pad" \
                --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
                --concurrency $CONC --limit $LIMIT --shuffle --out "$OUT" --raw "$RAW" \
                >> "$LOGD/retry_bench_${CELL}.log" 2>&1 || log "  bench FAIL $CELL x mix"
            log "  done $CELL"
        else
            log "  SKIP $CELL (boot fail again)"; tail -25 "$BL"
        fi
        kill_pg "$PID"; wait_gpu
    done
done
log "=== retry done ==="
