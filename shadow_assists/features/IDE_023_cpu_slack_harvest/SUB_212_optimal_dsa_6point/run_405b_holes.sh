#!/usr/bin/env bash
# SUB_212 빈 14셀 채우기 — 405B-FP8 의 ⑤ suf(ON) / ⑥ suf+dsa(ON), K=32, 7corpus×2
#   배경: §11.1 의 "suffix 부팅 한계" = K=32 일 때 (K+1)*conc capture > KV → num_gpu_blocks=0 크래시.
#         "영구 미측정" 단정은 과했음 (SUB_213 가 405B+suffix K4/6 정상부팅으로 반증).
#   전략: K=32 유지 + gmu 미세 스윕 {0.88,0.90,0.92} 로 num_gpu_blocks=0 벗어나는 최소 gmu 채택.
#         MML 16384·conc 32 canonical 유지. taskset 해제(전 코어), --allow-deprecated-quantization.
#   주의: gmu 가 SUB_212(0.85)와 다를 수 있음 → 셀별 채택 gmu 를 summ json/로그에 기록(=caveat).
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
GPUS=$(seq -s, 0 $((TP-1)))
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1
log(){ echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

# point|method_tag|extra_env
POINTS=(
  "5|suf|"
  "6|sufdsa|VLLM_LHC_DSA=1 VLLM_LEVER_N9=1"
)
GMUS=(0.88 0.90 0.92)

wait_ready(){ local i; for i in $(seq 1 600); do   # 50분
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "READY ($((i*5))s)"; return 0; }
    kill -0 "$1" 2>/dev/null || { log "DEAD backend"; return 1; }; sleep 5; done; log TIMEOUT; return 1; }
kill_pg(){ local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
wait_gpu(){ local i u; for i in $(seq 1 100); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}')
    [ "${u:-1}" -lt 4000 ] && return 0; sleep 5; done; return 0; }
trap 'echo "[trap] interrupted"; kill_pg ${PID:-}; exit 130' INT TERM

# pad sweep(run_retry_failed.sh) 가 끝날 때까지 대기 — GPU 충돌 방지
log "=== wait for pad sweep (run_retry_failed.sh / run_multimodel_pad.sh) to finish ==="
while pgrep -f 'run_retry_failed.sh|run_multimodel_pad.sh' >/dev/null 2>&1; do sleep 60; done
wait_gpu
log "=== pad sweep done → start 405B hole-fill (⑤⑥ K=32) ==="

for ENTRY in "${POINTS[@]}"; do
    IFS='|' read -r PT MTAG ENVX <<< "$ENTRY"
    METHOD="suffix_k32_pt${PT}_${MTAG}"
    # 이미 7 corpus + mix 다 있으면 skip
    DONE=$(ls "$OUTDIR"/summ_${TAG}_${METHOD}_*.json 2>/dev/null | wc -l)
    [ "$DONE" -ge 7 ] && { log "skip pt${PT} (done)"; continue; }
    BOOTED=0; USED_GMU=""
    for GMU in "${GMUS[@]}"; do
        BL="$LOGD/boot_pt${PT}_gmu${GMU}.log"; : > "$BL"
        log "boot pt${PT}(${MTAG}) gmu=$GMU K=$K"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES=$GPUS $ENVX setsid \
            "$VBIN" serve "$MODEL" --tensor-parallel-size $TP --port $PORT \
            --gpu-memory-utilization $GMU --max-model-len $MML \
            --allow-deprecated-quantization \
            --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
            --speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":$K}" \
            > "$BL" 2>&1 < /dev/null &
        PID=$!
        if wait_ready "$PID"; then BOOTED=1; USED_GMU=$GMU; break
        else log "  boot fail gmu=$GMU (num_gpu_blocks=0?)"; grep -i 'num_gpu_blocks\|no available memory\|out of memory' "$BL" | tail -3
             kill_pg "$PID"; wait_gpu; fi
    done
    if [ "$BOOTED" != 1 ]; then log "  GIVE UP pt${PT} (all gmu failed)"; continue; fi
    log "  booted pt${PT} at gmu=$USED_GMU → bench"
    for C in "${CORPORA[@]}"; do
        OUT="$OUTDIR/summ_${TAG}_${METHOD}_${C}.json"
        [ -s "$OUT" ] && continue
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$METHOD" \
            --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
            --concurrency $CONC --corpus "$C" --out "$OUT" --raw "$RAW" \
            >> "$LOGD/bench_pt${PT}.log" 2>&1 || log "  bench FAIL pt${PT} x $C"
    done
    OUT="$OUTDIR/summ_${TAG}_${METHOD}_mix.json"
    [ -s "$OUT" ] || PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$METHOD" \
        --model "$MODEL" --model-tag "$TAG" --port $PORT --max-tokens $MAXTOK \
        --concurrency $CONC --limit $LIMIT --shuffle --out "$OUT" --raw "$RAW" \
        >> "$LOGD/bench_pt${PT}.log" 2>&1 || log "  bench FAIL pt${PT} x mix"
    echo "$USED_GMU" > "$OUTDIR/gmu_used_pt${PT}.txt"
    log "  done pt${PT} (gmu=$USED_GMU)"
    kill_pg "$PID"; wait_gpu
done
log "=== 405B hole-fill done ==="
