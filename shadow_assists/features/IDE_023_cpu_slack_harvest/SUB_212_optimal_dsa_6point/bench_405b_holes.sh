#!/usr/bin/env bash
# SUB_212 405B ⑤⑥ K32 hole-fill — bench (max-num-seqs 32 로 capture 바운드해 부팅 성공)
#   PT5 = 이미 떠 있는 ⑤ 서버(port 8011)를 벤치 → kill → PT6 부팅·벤치.
#   ⑤ = suffix K32 + FaP, host DSA on(system), pad OFF. ⑥ = + VLLM_LHC_DSA=1 VLLM_LEVER_N9=1.
#   canonical 유지: gmu 0.85, MML 16384, conc 32, max_tok 8192. 추가만 --max-num-seqs 32 (capture 바운드).
set -u
cd /home/mystous/vllm_hybrid
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
SUBD=shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_212_optimal_dsa_6point
OUTDIR=$SUBD/runs_405b_holes; LOGD="$OUTDIR/_logs"; mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"
SAMPLED="$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet"
CORPORA=(sharegpt swebench humaneval mbpp wildchat lmsys)
PORT=8011 MML=16384 CONC=32 MAXTOK=8192 LIMIT=500 K=32
MODEL=meta-llama/Llama-3.1-405B-Instruct-FP8 TAG=Llama-3.1-405B-Instruct-FP8 TP=8
GPUS=0,1,2,3,4,5,6,7
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1
log(){ echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }
wait_ready(){ local i; for i in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "READY ($((i*5))s)"; return 0; }
    kill -0 "$1" 2>/dev/null || { log "DEAD backend"; return 1; }; sleep 5; done; log TIMEOUT; return 1; }
kill_pg(){ local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
wait_gpu(){ local i u; for i in $(seq 1 100); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}')
    [ "${u:-1}" -lt 4000 ] && return 0; sleep 5; done; return 0; }
bench_all(){ local METHOD=$1
    for C in "${CORPORA[@]}"; do
        OUT="$OUTDIR/summ_${TAG}_${METHOD}_${C}.json"; [ -s "$OUT" ] && continue
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$METHOD" \
            --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
            --concurrency $CONC --corpus "$C" --out "$OUT" --raw "$RAW" \
            >> "$LOGD/bench_${METHOD}.log" 2>&1 || log "  bench FAIL $METHOD x $C"
    done
    OUT="$OUTDIR/summ_${TAG}_${METHOD}_mix.json"
    [ -s "$OUT" ] || PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$METHOD" \
        --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
        --concurrency $CONC --limit $LIMIT --shuffle --out "$OUT" --raw "$RAW" \
        >> "$LOGD/bench_${METHOD}.log" 2>&1 || log "  bench FAIL $METHOD x mix"; }

# ── PT5: 이미 떠있는 ⑤ 서버 벤치 (test_pt5_maxseqs32 부팅) ──
log "=== PT5 (suf ON, K32) bench — 기존 서버 재사용 ==="
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    bench_all "suffix_k32_pt5_suf"; log "  PT5 done"
    # 기존 ⑤ 서버 종료
    PG=$(pgrep -f "vllm serve.*405B.*num_speculative_tokens.:32" | head -1)
    [ -n "$PG" ] && kill_pg "$PG"; wait_gpu
else
    log "  PT5 서버 미가동 → skip (별도 부팅 필요)"
fi

# ── PT6: ⑥ suf+dsa(ON) 부팅·벤치 ──
log "=== PT6 (suf+dsa ON, K32) boot+bench ==="
DONE=$(ls "$OUTDIR"/summ_${TAG}_suffix_k32_pt6_sufdsa_*.json 2>/dev/null | wc -l)
if [ "$DONE" -lt 7 ]; then
    BL="$LOGD/boot_pt6_maxseqs32.log"; : > "$BL"
    env CUDA_VISIBLE_DEVICES=$GPUS VLLM_LHC_DSA=1 VLLM_LEVER_N9=1 setsid \
        "$VBIN" serve "$MODEL" --tensor-parallel-size $TP --port $PORT \
        --gpu-memory-utilization 0.85 --max-model-len $MML --max-num-seqs 32 \
        --allow-deprecated-quantization \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        --speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":$K}" \
        > "$BL" 2>&1 < /dev/null &
    PID=$!; log "boot PT6 (pid=$PID)"
    if wait_ready "$PID"; then bench_all "suffix_k32_pt6_sufdsa"; log "  PT6 done"
    else log "  PT6 boot fail"; grep -iE 'num_gpu_blocks|Available KV|Error' "$BL" | tail -5; fi
    kill_pg "$PID"; wait_gpu
else log "  PT6 already done"; fi
log "=== 405B ⑤⑥ hole-fill bench done ==="
