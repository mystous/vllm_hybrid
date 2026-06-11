#!/usr/bin/env bash
# Optimal+DSA real-corpus validation — fills the 4-version comparison
# (Vanilla / DSA / Optimal=suffix / Optimal+DSA) on the SAME real trace
# corpus that TSK_042 used. Direct cell-by-cell comparability.
#
# Mirrors realistic_eval/run_throughput_8gpu.sh:
#   - TP=8, gpu_mem=0.85, max_model_len=16384
#   - cudagraph_mode=FULL_AND_PIECEWISE (FaP) — TSK_042 setting
#   - conc=32, max_tokens=8192, streaming (TTFT/TPOT split)
#   - 6 isolated corpus + mix (shuffle 500)
#
# Configs:
#   vanilla      — baseline
#   dsa          — VLLM_LHC_DSA=1 + VLLM_LEVER_N9=1 (DSA lane on)
#   suffix       — speculative-config suffix K=32 (TSK_042 dominant)
#   suffix_dsa   — suffix + DSA  ← KEY CELL (Optimal+DSA)
#
# n=1 per cell (per user request "한 번씩만 돌려").
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TAG="Llama-3.1-8B-Instruct"
TP=8
GPUS=0,1,2,3,4,5,6,7
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
: > "$RAW"

# Required env per TSK_042 harness.
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready(){
    local i
    for i in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "READY $1"; return 0; }
        [ -n "${2:-}" ] && ! kill -0 "$2" 2>/dev/null && { log "DEAD backend (boot 실패) $1"; return 1; }
        sleep 5
    done
    log "TIMEOUT $1"; return 1
}

kill_pgroup(){
    local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
    kill -9 "$pid" 2>/dev/null
    # also orphans
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

trap 'echo "[trap] interrupted at $(date -u +%FT%TZ)"; kill_pgroup ${PID:-}; exit 130' INT TERM

log "=== Optimal+DSA real-corpus sweep start ==="
log "model=$MODEL TP=$TP corpora=[${CORPORA[*]} mix] conc=$CONC maxtok=$MAXTOK"
wait_gpu_free || true

CONFIGS=(vanilla dsa suffix suffix_dsa)

for CFG in "${CONFIGS[@]}"; do
    log "### config=$CFG ###"
    SA=$(spec_args "$CFG")
    ENV_PRE=$(extra_env_for "$CFG")

    BOOT_LOG="$LOGD/${CFG}_boot.log"
    : > "$BOOT_LOG"

    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=$GPUS $ENV_PRE setsid "$VBIN" serve "$MODEL" \
        --tensor-parallel-size $TP --port $PORT \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MML" \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        $SA > "$BOOT_LOG" 2>&1 < /dev/null &
    PID=$!
    log "boot pid=$PID for $CFG (env: $ENV_PRE)"

    if wait_ready "http://127.0.0.1:$PORT" "$PID"; then
        for C in "${CORPORA[@]}"; do
            OUT="$OUTDIR/summ_${TAG}_${CFG}_${C}.json"
            if [ -s "$OUT" ]; then
                log "  skip $CFG x $C (exists)"
                continue
            fi
            log "  bench $CFG x $C"
            PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
                --in "$SAMPLED" --method "$CFG" \
                --model "$MODEL" --model-tag "$TAG" \
                --port $PORT --max-tokens "$MAXTOK" \
                --concurrency "$CONC" --corpus "$C" \
                --out "$OUT" --raw "$RAW" \
                >> "$LOGD/${CFG}_bench.log" 2>&1 || log "    bench fail $CFG x $C"
        done
        # mix (shuffle 500)
        OUT="$OUTDIR/summ_${TAG}_${CFG}_mix.json"
        if [ ! -s "$OUT" ]; then
            log "  bench $CFG x mix"
            PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
                --in "$SAMPLED" --method "$CFG" \
                --model "$MODEL" --model-tag "$TAG" \
                --port $PORT --max-tokens "$MAXTOK" \
                --concurrency "$CONC" --limit "$LIMIT" --shuffle \
                --out "$OUT" --raw "$RAW" \
                >> "$LOGD/${CFG}_bench.log" 2>&1 || log "    bench fail $CFG x mix"
        fi
        log "  done $CFG (7 cells)"
    else
        log "  SKIP $CFG (boot 실패)"
        tail -60 "$BOOT_LOG"
    fi
    kill_pgroup "$PID"
    wait_gpu_free
done

log "=== sweep complete ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -10
