#!/usr/bin/env bash
# verify_fap.sh — SUB_212 host-DSA confounder 결론의 재검증 (FaP 가설)
#
# 발견 (2026-06-11):
#   TSK_042 boot log  : cudagraph_mode=PIECEWISE          (06-02, van 8,850 tps)
#   SUB_212 boot logs : cudagraph_mode=FULL_AND_PIECEWISE (06-10, van 12,089 tps)
#   sweep_corpus.sh:120 가 모든 부팅에 FaP 를 명시 전달 (주석은 "TSK_042 setting" 으로 오기)
#   + host DSA WQ clients=0 (vanilla 측정 중 아무 프로세스도 WQ 미사용)
#
# 가설 H-FaP: vanilla +36% 의 진짜 원인 = FaP (host DSA WQ 무관)
# 검증: 현 환경 (host WQ enabled 그대로) 에서 PIECEWISE 로만 부팅
#   E1 vanilla+PIECEWISE  mix → 예측 ~8,850  (TSK_042 재현 시 H-FaP 확정)
#   E2 suffix +PIECEWISE  mix → 예측 ~27,851 (suffix OFF→ON "−12%" 도 FaP 효과로 재해석)
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

OUTDIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_212_optimal_dsa_6point/runs_verify_fap
LOGD="$OUTDIR/_logs"
mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"
touch "$RAW"

# TSK_042 harness 와 동일 env
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
        vanilla) echo "";;
        suffix)  echo '--speculative-config {"method":"suffix","num_speculative_tokens":32}' ;;
    esac
}

trap 'echo "[trap] interrupted at $(date -u +%FT%TZ)"; kill_pgroup ${PID:-}; exit 130' INT TERM

log "=== verify_fap: PIECEWISE A/B on current env (host WQ enabled 그대로) ==="
wait_gpu_free || true

for CFG in vanilla suffix; do
    OUT="$OUTDIR/summ_${TAG}_${CFG}_piecewise_mix.json"
    if [ -s "$OUT" ]; then log "skip $CFG (exists)"; continue; fi

    log "### E: ${CFG}+PIECEWISE ###"
    SA=$(spec_args "$CFG")
    BOOT_LOG="$LOGD/${CFG}_piecewise_boot.log"
    : > "$BOOT_LOG"

    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=$GPUS setsid "$VBIN" serve "$MODEL" \
        --tensor-parallel-size $TP --port $PORT \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MML" \
        --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
        $SA > "$BOOT_LOG" 2>&1 < /dev/null &
    PID=$!
    log "boot pid=$PID for ${CFG}+PIECEWISE"

    if wait_ready "http://127.0.0.1:$PORT" "$PID"; then
        log "  bench ${CFG}+PIECEWISE x mix"
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
            --in "$SAMPLED" --method "${CFG}_piecewise" \
            --model "$MODEL" --model-tag "$TAG" \
            --port $PORT --max-tokens "$MAXTOK" \
            --concurrency "$CONC" --limit "$LIMIT" --shuffle \
            --out "$OUT" --raw "$RAW" \
            >> "$LOGD/${CFG}_piecewise_bench.log" 2>&1 || log "  bench fail"
        log "  done ${CFG}+PIECEWISE"
    else
        log "  SKIP ${CFG}+PIECEWISE (boot 실패)"; tail -40 "$BOOT_LOG"
    fi
    kill_pgroup "$PID"
    wait_gpu_free
done

log "=== verify_fap complete ==="
for f in "$OUTDIR"/summ_*.json; do
    echo "--- $f"; "$PY" - "$f" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print({k:d.get(k) for k in ("method","corpus","gen_tps","total_tps","ttft_p50_ms","tpot_p50_ms") if k in d} or list(d)[:12])
EOF
done
