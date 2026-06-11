#!/usr/bin/env bash
# Complete 6/6 coverage sweep — fill 154 missing cells.
#
# Existing coverage (호스트 DSA enabled, 2026-06-10 sweep):
#   - Llama-3.1-8B-Instruct: 6/6 ✅ (모든 점)
#   - 7 mid models (Qwen-7B, DS-Qwen-7B, Qwen-32B, DS-Qwen-32B, Qwen-72B,
#     Llama-70B, DS-Llama-70B): 4/6 (dsa + suffix_dsa 만)
#   - 2 XL models (Llama-405B-FP8, DeepSeek-R1): 2/6 (TSK_042 OFF baseline 만)
#
# This sweep adds (호스트 DSA 활성 상태 유지):
#   - 7 mid models × {vanilla, suffix} × 7 corpus = 98 cells
#   - 2 XL models × {vanilla, dsa, suffix, suffix_dsa} × 7 corpus = 56 cells
#   Total: 154 cells
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

# (HF_NAME | TAG | TP | configs to measure | extra cli flags)
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|4|vanilla suffix|"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|DeepSeek-R1-Distill-Qwen-7B|4|vanilla suffix|"
    "Qwen/Qwen2.5-32B-Instruct|Qwen2.5-32B-Instruct|8|vanilla suffix|"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|DeepSeek-R1-Distill-Qwen-32B|8|vanilla suffix|"
    "Qwen/Qwen2.5-72B-Instruct|Qwen2.5-72B-Instruct|8|vanilla suffix|"
    "meta-llama/Llama-3.1-70B-Instruct|Llama-3.1-70B-Instruct|8|vanilla suffix|"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B|DeepSeek-R1-Distill-Llama-70B|8|vanilla suffix|"
    "meta-llama/Llama-3.1-405B-Instruct-FP8|Llama-3.1-405B-Instruct-FP8|8|vanilla dsa suffix suffix_dsa|--allow-deprecated-quantization"
    "deepseek-ai/DeepSeek-R1|DeepSeek-R1|8|vanilla dsa suffix suffix_dsa|"
)

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready(){
    local i
    for i in $(seq 1 360); do  # 30min 까지 wait (XL 모델 부팅 위해)
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
            && { log "READY"; return 0; }
        [ -n "${1:-}" ] && ! kill -0 "$1" 2>/dev/null && { log "DEAD backend"; return 1; }
        sleep 5
    done
    log "TIMEOUT"; return 1
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
    for i in $(seq 1 90); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk '{s+=$1}END{print s+0}')
        [ "${u:-1}" -lt 4000 ] && { log "  gpu freed"; return 0; }
        sleep 5
    done
    log "  GPU NOT FREED"; return 0
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
trap 'log "[trap] interrupted at $(date -u +%FT%TZ)"; kill_pgroup ${PID:-}; exit 130' INT TERM

log "=== Complete 6/6 coverage sweep start ==="
log "Host DSA WQ state: $(cat /sys/bus/dsa/devices/wq0.0/state)"
log "models=${#MODELS[@]} corpora=7+mix"
wait_gpu_free || true

for ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r MODEL TAG TP CONFIGS_STR EXTRA_FLAGS <<< "$ENTRY"
    read -r -a CONFIGS_LIST <<< "$CONFIGS_STR"

    if [ "$TP" = 8 ]; then GPUS=0,1,2,3,4,5,6,7
    else GPUS=0,1,2,3
    fi

    log "########## model=$TAG (TP=$TP) configs=[${CONFIGS_STR}] ##########"

    for CFG in "${CONFIGS_LIST[@]}"; do
        # 이미 모두 측정됐는지 확인
        DONE=0
        for C in "${CORPORA[@]}" mix; do
            [ -s "$OUTDIR/summ_${TAG}_${CFG}_${C}.json" ] && DONE=$((DONE + 1))
        done
        if [ "$DONE" -eq 7 ]; then
            log "  skip $CFG (all 7 cells exist)"
            continue
        fi

        log "  ### config=$CFG ($DONE/7 done) ###"
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
            $EXTRA_FLAGS \
            $SA > "$BOOT_LOG" 2>&1 < /dev/null &
        PID=$!
        log "  boot pid=$PID for $TAG/$CFG (env: $ENV_PRE | flags: $EXTRA_FLAGS)"

        if wait_ready "$PID"; then
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
            log "  SKIP $TAG/$CFG (boot failed)"
            tail -60 "$BOOT_LOG"
        fi
        kill_pgroup "$PID"
        wait_gpu_free
    done
    log "########## done $TAG ##########"
done

log "=== Complete sweep finished ==="
ls "$OUTDIR" | grep -c "^summ_.*\.json$" || true
