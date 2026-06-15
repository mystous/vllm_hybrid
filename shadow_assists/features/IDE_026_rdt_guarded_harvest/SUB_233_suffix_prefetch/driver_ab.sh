#!/usr/bin/env bash
# SUB_233 A/B — 70B suffix pad(K6) baseline vs prefetch (.so 교체 + fresh boot).
#   arm 마다: .so 설치 → 70B 부팅 → bench(mix,swebench,lmsys) → tps 기록 → kill.
#   baseline/prefetch 모두 내가 동일 플래그로 빌드(컴파일러/nanobind 동일, prefetch 코드만 차이).
#   끝나면 원본 .so(.orig) 복원.
set -u
cd /home/mystous/vllm_hybrid
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
FD=shadow_assists/features/IDE_026_rdt_guarded_harvest/SUB_233_suffix_prefetch
OUTDIR=$FD/runs; LOGD=$OUTDIR/_logs; mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"
SAMPLED="$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet"
SO_INSTALL=/home/mystous/vllm_dev_prj/lib/python3.12/site-packages/arctic_inference/suffix_decoding/_C.cpython-312-x86_64-linux-gnu.so
CORPORA=(mix swebench lmsys)
PORT=8011 MML=16384 CONC=32 MAXTOK=8192 LIMIT=500 K=6
VLLM_CPUS="0-47,56-103"   # fullmatrix baseline 과 동일 taskset (교란 제거)
MODEL=meta-llama/Llama-3.1-70B-Instruct TAG=Llama-3.1-70B-Instruct TP=8
GPUS=$(seq -s, 0 $((TP-1)))
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1
ARMS="$@"   # 예: "baseline prefetch" 또는 "prefetch baseline prefetch"
log(){ echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }
wait_ready(){ local i; for i in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "READY ($((i*5))s)"; return 0; }
    kill -0 "$1" 2>/dev/null || { log "DEAD backend"; return 1; }; sleep 5; done; log TIMEOUT; return 1; }
kill_pg(){ local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
wait_gpu(){ local i u; for i in $(seq 1 100); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}')
    [ "${u:-1}" -lt 4000 ] && return 0; sleep 5; done; return 0; }
trap 'echo "[trap] interrupted"; kill_pg ${PID:-}; cp "$FD/_C_orig.so" "$SO_INSTALL" 2>/dev/null; exit 130' INT TERM

# 원본 백업 보존
[ -f "$FD/_C_orig.so" ] || cp "${SO_INSTALL}.orig" "$FD/_C_orig.so" 2>/dev/null || cp "$SO_INSTALL" "$FD/_C_orig.so"

run_arm(){ local ARM=$1 RUNTAG=$2
    log "=== ARM=$ARM ($RUNTAG): install .so + boot ==="
    cp "$FD/_C_${ARM}.so" "$SO_INSTALL" || { log "so copy fail"; return 1; }
    BL="$LOGD/boot_${RUNTAG}.log"; : > "$BL"
    env CUDA_VISIBLE_DEVICES=$GPUS VLLM_SUFFIX_PAD_UNIFORM=1 setsid taskset -c "$VLLM_CPUS" \
        "$VBIN" serve "$MODEL" --tensor-parallel-size $TP --port $PORT \
        --gpu-memory-utilization 0.85 --max-model-len $MML \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        --speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":$K}" \
        > "$BL" 2>&1 < /dev/null &
    PID=$!; log "boot $RUNTAG (pid=$PID)"
    if wait_ready "$PID"; then
        for C in "${CORPORA[@]}"; do
            OUT="$OUTDIR/summ_${RUNTAG}_${C}.json"
            if [ "$C" = "mix" ]; then
                PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$RUNTAG" \
                    --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
                    --concurrency $CONC --limit $LIMIT --shuffle --out "$OUT" --raw "$RAW" \
                    >> "$LOGD/bench_${RUNTAG}.log" 2>&1 || log "  bench FAIL $RUNTAG x $C"
            else
                PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$RUNTAG" \
                    --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
                    --concurrency $CONC --corpus "$C" --out "$OUT" --raw "$RAW" \
                    >> "$LOGD/bench_${RUNTAG}.log" 2>&1 || log "  bench FAIL $RUNTAG x $C"
            fi
            V=$($PY -c "import json;print(f\"{json.load(open('$OUT'))['output_tps']:.1f}\")" 2>/dev/null || echo "NA")
            log "  $RUNTAG x $C tps=$V"
        done
    else log "  BOOT FAIL $RUNTAG"; tail -20 "$BL"; fi
    kill_pg "$PID"; wait_gpu
}

log "=== SUB_233 A/B start: arms=[$ARMS] ==="
i=0
for ARM in $ARMS; do
    i=$((i+1)); run_arm "$ARM" "r${i}_${ARM}"
done
# 원본 복원
cp "$FD/_C_orig.so" "$SO_INSTALL" && log "원본 .so 복원"
log "=== SUB_233 A/B done ==="
