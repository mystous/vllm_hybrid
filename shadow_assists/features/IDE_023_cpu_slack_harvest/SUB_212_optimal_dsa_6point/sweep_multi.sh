#!/usr/bin/env bash
# Optimal+DSA multi-model real-corpus validation — extends Llama-3.1-8B
# (already done) to all 8 cached models (R1 671B deferred — needs 2×TP4
# setup).
#
# Mirrors realistic_eval/run_throughput_8gpu.sh structure:
#   - TP picked per model (head%8==0 → TP=8, else TP=4)
#   - cudagraph_mode=FULL_AND_PIECEWISE (FaP)
#   - conc=32, max_tokens=8192, streaming
#   - 6 isolated corpus + mix (shuffle 500)
#   - n=1 per cell (per user request)
#
# 4 configs × 7 corpus × 8 models = 224 cells.
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval

PORT=8001
MML=16384
CONC=32
MAXTOK=8192
LIMIT=500
SAMPLED="$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet"
CORPORA=(sharegpt swebench humaneval mbpp wildchat lmsys)

OUTDIR=/workspace/host_vllm_hybrid/lhc_phase4/optimal_dsa/runs
LOGD="$OUTDIR/_logs"
mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"
touch "$RAW"

# Models smallest → largest. Already-done Llama-8B included so re-uses
# its 7 cells via the "skip existing" guard.
# Format: "HF_NAME|TAG|TP"
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|4"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|DeepSeek-R1-Distill-Qwen-7B|4"
    "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct|8"
    "Qwen/Qwen2.5-32B-Instruct|Qwen2.5-32B-Instruct|8"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|DeepSeek-R1-Distill-Qwen-32B|8"
    "Qwen/Qwen2.5-72B-Instruct|Qwen2.5-72B-Instruct|8"
    "meta-llama/Llama-3.1-70B-Instruct|Llama-3.1-70B-Instruct|8"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B|DeepSeek-R1-Distill-Llama-70B|8"
)

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready(){
    local i
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
            && { log "READY $1"; return 0; }
        [ -n "${2:-}" ] && ! kill -0 "$2" 2>/dev/null \
            && { log "DEAD backend (boot 실패) $1"; return 1; }
        sleep 5
    done
    log "TIMEOUT $1"; return 1
}

kill_pgroup(){
    local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
    kill -9 "$pid" 2>/dev/null
    for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
        [ -d "/proc/$op" ] && kill -9 "$op" 2>/dev/null
    done
}

wait_gpu_free(){
    local i u
    for i in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk '{s+=$1}END{print s+0}')
        [ "${u:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }
        sleep 5
    done
    log "GPU NOT FREED"; return 0
}

spec_args(){
    case "$1" in
        vanilla|dsa) echo "";;
        suffix|suffix_dsa) echo '--speculative-config {"method":"suffix","num_speculative_tokens":32}' ;;
        *) echo "";;
    esac
}

extra_env_for(){
    case "$1" in
        vanilla|suffix) echo "";;
        dsa|suffix_dsa) echo "VLLM_LHC_DSA=1 VLLM_LEVER_N9=1 VLLM_LHC_DSA_MIN=65536" ;;
        *) echo "";;
    esac
}

PID=""
trap 'echo "[trap] interrupted at $(date -u +%FT%TZ)"; kill_pgroup ${PID:-}; exit 130' INT TERM

log "=== Optimal+DSA multi-model sweep start ==="
log "models=${#MODELS[@]} configs=4 corpora=7 (n=1)"
wait_gpu_free || true

CONFIGS=(dsa suffix_dsa)  # vanilla/suffix는 TSK_042 기존 측정값 사용

for ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL TAG TP <<< "$ENTRY"
    if [ "$TP" = 8 ]; then GPUS=0,1,2,3,4,5,6,7
    else GPUS=0,1,2,3
    fi

    # Skip model entirely if all needed cells already exist (2 configs × 7 corpus).
    DONE=0
    EXPECTED=$((${#CONFIGS[@]} * (${#CORPORA[@]} + 1)))
    for CFG in "${CONFIGS[@]}"; do
        for C in "${CORPORA[@]}" mix; do
            [ -s "$OUTDIR/summ_${TAG}_${CFG}_${C}.json" ] && DONE=$((DONE + 1))
        done
    done
    if [ "$DONE" -eq "$EXPECTED" ]; then
        log "########## SKIP $TAG (all $EXPECTED cells exist) ##########"
        continue
    fi

    log "########## model=$TAG (TP=$TP, $DONE/$EXPECTED done) ##########"

    for CFG in "${CONFIGS[@]}"; do
        # Check this config's cells already done
        CFG_DONE=0
        for C in "${CORPORA[@]}" mix; do
            [ -s "$OUTDIR/summ_${TAG}_${CFG}_${C}.json" ] && CFG_DONE=$((CFG_DONE + 1))
        done
        if [ "$CFG_DONE" -eq $((${#CORPORA[@]} + 1)) ]; then
            log "  skip config $CFG (all 7 cells exist)"
            continue
        fi

        log "  ### config=$CFG ($CFG_DONE/7 done) ###"
        SA=$(spec_args "$CFG")
        ENV_PRE=$(extra_env_for "$CFG")

        BOOT_LOG="$LOGD/${TAG}_${CFG}_boot.log"
        : > "$BOOT_LOG"

        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES=$GPUS $ENV_PRE setsid "$VBIN" serve "$MODEL" \
            --tensor-parallel-size $TP --port $PORT \
            --gpu-memory-utilization 0.85 \
            --max-model-len "$MML" \
            --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
            $SA > "$BOOT_LOG" 2>&1 < /dev/null &
        PID=$!
        log "  boot pid=$PID for $TAG/$CFG"

        if wait_ready "http://127.0.0.1:$PORT" "$PID"; then
            for C in "${CORPORA[@]}"; do
                OUT="$OUTDIR/summ_${TAG}_${CFG}_${C}.json"
                if [ -s "$OUT" ]; then
                    log "    skip $TAG/$CFG/$C (exists)"
                    continue
                fi
                log "    bench $TAG/$CFG/$C"
                PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
                    --in "$SAMPLED" --method "$CFG" \
                    --model "$MODEL" --model-tag "$TAG" \
                    --port $PORT --max-tokens "$MAXTOK" \
                    --concurrency "$CONC" --corpus "$C" \
                    --out "$OUT" --raw "$RAW" \
                    >> "$LOGD/${TAG}_${CFG}_bench.log" 2>&1 \
                    || log "      bench fail $TAG/$CFG/$C"
            done
            OUT="$OUTDIR/summ_${TAG}_${CFG}_mix.json"
            if [ ! -s "$OUT" ]; then
                log "    bench $TAG/$CFG/mix"
                PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
                    --in "$SAMPLED" --method "$CFG" \
                    --model "$MODEL" --model-tag "$TAG" \
                    --port $PORT --max-tokens "$MAXTOK" \
                    --concurrency "$CONC" --limit "$LIMIT" --shuffle \
                    --out "$OUT" --raw "$RAW" \
                    >> "$LOGD/${TAG}_${CFG}_bench.log" 2>&1 \
                    || log "      bench fail $TAG/$CFG/mix"
            fi
            log "  done $TAG/$CFG"
        else
            log "  SKIP $TAG/$CFG (boot 실패)"
            tail -40 "$BOOT_LOG"
        fi
        kill_pgroup "$PID"
        wait_gpu_free
    done
    log "########## done $TAG ##########"
done

log "=== multi-model sweep complete ==="
ls "$OUTDIR" | grep -c "^summ_.*\.json$" || true
