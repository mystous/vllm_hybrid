#!/usr/bin/env bash
# sub213_sweep.sh — SUB_213 FaP×suffix 양립 lever (Uniform Draft Padding) P1~P4
#
# 셀 설계 (Llama-8B, mix, conc=32):
#   P1  K=8  pad   FaP        — 본명제: FULL graph 적중 시 suffix 위 FaP 가산
#   P2  K=15 pad   FaP        — K 상한 (32×16=512 정확) trade-off
#   P3  K=8  nopad FaP        — K 축소 단독 효과 (FULL 미적중 대조군)
#   P4  K=8  nopad PIECEWISE  — K 축소 + PIECEWISE 기준점
#
# 기준점: suf K=32 PIECEWISE = 27,851 (TSK_042) / suf K=32 FaP = 24,407 (SUB_212)
# 사전 예측 (IDE_023 P0): P1 = +10~25% vs 27,851 → GO / P1 < 27,851 → kill / P3 ≈ P4
#
# pad lever 는 env VLLM_SUFFIX_PAD_UNIFORM=1 + flag file
# /tmp/vllm_l3_VLLM_SUFFIX_PAD_UNIFORM.flag (EngineCore spawn 이 env 를 못 받는
# 경우 대비) 둘 다 세팅. nopad 셀에서는 둘 다 반드시 제거 (오염 방지).
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

SUBDIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform
OUTDIR="$SUBDIR/runs"
LOGD="$OUTDIR/_logs"
mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"
touch "$RAW"

PAD_FLAG=/tmp/vllm_l3_VLLM_SUFFIX_PAD_UNIFORM.flag

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

# cell <name> <K> <pad:1|0> <cudagraph_mode>
run_cell(){
    local NAME=$1 K=$2 PAD=$3 CGMODE=$4
    local OUT="$OUTDIR/summ_${TAG}_${NAME}_mix.json"
    if [ -s "$OUT" ]; then log "skip $NAME (exists)"; return 0; fi

    log "### $NAME: K=$K pad=$PAD cudagraph=$CGMODE ###"

    if [ "$PAD" = "1" ]; then
        export VLLM_SUFFIX_PAD_UNIFORM=1
        echo 1 > "$PAD_FLAG"
    else
        unset VLLM_SUFFIX_PAD_UNIFORM
        rm -f "$PAD_FLAG"
    fi

    local BOOT_LOG="$LOGD/${NAME}_boot.log"
    : > "$BOOT_LOG"

    env CUDA_VISIBLE_DEVICES=$GPUS setsid "$VBIN" serve "$MODEL" \
        --tensor-parallel-size $TP --port $PORT \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MML" \
        --compilation-config "{\"cudagraph_mode\":\"$CGMODE\"}" \
        --speculative-config "{\"method\":\"suffix\",\"num_speculative_tokens\":$K}" \
        > "$BOOT_LOG" 2>&1 < /dev/null &
    PID=$!
    log "boot pid=$PID for $NAME"

    if wait_ready "http://127.0.0.1:$PORT" "$PID"; then
        log "  bench $NAME x mix"
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
            --in "$SAMPLED" --method "$NAME" \
            --model "$MODEL" --model-tag "$TAG" \
            --port $PORT --max-tokens "$MAXTOK" \
            --concurrency "$CONC" --limit "$LIMIT" --shuffle \
            --out "$OUT" --raw "$RAW" \
            >> "$LOGD/${NAME}_bench.log" 2>&1 || log "  bench fail"
        log "  done $NAME"
    else
        log "  SKIP $NAME (boot 실패)"; tail -40 "$BOOT_LOG"
    fi
    kill_pgroup "$PID"
    # 다음 셀 오염 방지 — pad 상태는 셀 진입 시 다시 세팅됨
    unset VLLM_SUFFIX_PAD_UNIFORM
    rm -f "$PAD_FLAG"
    wait_gpu_free
}

trap 'echo "[trap] interrupted at $(date -u +%FT%TZ)"; rm -f "$PAD_FLAG"; kill_pgroup ${PID:-}; exit 130' INT TERM

log "=== sub213_sweep: P1~P4 (uniform draft padding × cudagraph) ==="
wait_gpu_free || true

#        name             K  pad cudagraph
run_cell suffix_k8_pad_fap     8  1 FULL_AND_PIECEWISE
run_cell suffix_k15_pad_fap   15  1 FULL_AND_PIECEWISE
run_cell suffix_k8_nopad_fap   8  0 FULL_AND_PIECEWISE
run_cell suffix_k8_nopad_pw    8  0 PIECEWISE

log "=== sub213_sweep complete ==="
for f in "$OUTDIR"/summ_*.json; do
    [ -s "$f" ] || continue
    echo "--- $f"; "$PY" - "$f" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print({k:d.get(k) for k in ("method","corpus","gen_tps","total_tps","ttft_p50_ms","tpot_p50_ms") if k in d} or list(d)[:12])
EOF
done
