#!/usr/bin/env bash
# C4: 호스트 DSA WQ disable + vanilla 측정.
# 가설: DSA WQ disabled → TSK_042 vanilla 의 8,850 tps 근처로 복원
# 반박: 12,089 그대로 → DSA 가 원인 아니고 다른 변수
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TAG="Llama-3.1-8B-Instruct"
PORT=8001
TP=8
GPUS=0,1,2,3,4,5,6,7
MML=16384
CONC=32
MAXTOK=8192
LIMIT=500
SAMPLED="$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet"
OUTDIR=/workspace/host_vllm_hybrid/lhc_phase4/optimal_dsa/runs
LOGD="$OUTDIR/_logs"
mkdir -p "$LOGD"

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# Save current WQ states for restore
declare -a WQ_LIST=(wq0.0 wq0.1 wq0.2 wq0.3 wq1.0 wq1.1 wq1.2 wq1.3)
declare -a ORIG_STATE
PID=""
DSA_MODIFIED=0

restore_dsa(){
    if [ "$DSA_MODIFIED" -eq 1 ]; then
        log "[restore] DSA WQ states"
        local i=0
        for wq in "${WQ_LIST[@]}"; do
            local orig=${ORIG_STATE[$i]}
            if [ "$orig" = "enabled" ]; then
                echo enabled > /sys/bus/dsa/devices/$wq/state 2>/dev/null \
                    && log "  $wq → enabled" \
                    || log "  $wq restore FAIL"
            fi
            i=$((i+1))
        done
        DSA_MODIFIED=0
    fi
}

cleanup(){
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        local pg; pg=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
        [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
        kill -9 "$PID" 2>/dev/null
    fi
    restore_dsa
}
trap 'log "[trap] interrupted"; cleanup; exit 130' INT TERM EXIT

log "=== verify_host_dsa start ==="

# Save current WQ states
log "Save current WQ states"
i=0
for wq in "${WQ_LIST[@]}"; do
    s=$(cat /sys/bus/dsa/devices/$wq/state 2>/dev/null || echo "missing")
    ORIG_STATE[$i]=$s
    log "  $wq: $s"
    i=$((i+1))
done

# Disable enabled WQs
log "Disable all enabled WQs"
i=0
for wq in "${WQ_LIST[@]}"; do
    if [ "${ORIG_STATE[$i]}" = "enabled" ]; then
        echo disabled > /sys/bus/dsa/devices/$wq/state 2>/dev/null \
            && log "  $wq → disabled" \
            || log "  $wq disable FAIL"
        DSA_MODIFIED=1
    fi
    i=$((i+1))
done

# Verify
log "Post-disable state"
for wq in "${WQ_LIST[@]}"; do
    s=$(cat /sys/bus/dsa/devices/$wq/state 2>/dev/null)
    log "  $wq: $s"
done

# Boot vllm vanilla
log "Boot vllm (C4_hostDSA_disabled)"
BOOT_LOG=$LOGD/C4_hostDSA_disabled_boot.log
: > "$BOOT_LOG"
env CUDA_VISIBLE_DEVICES=$GPUS setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size $TP --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len "$MML" \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
log "  pid=$PID"

# Wait for ready
for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "  READY"; break; }
    ! kill -0 "$PID" 2>/dev/null && { log "  DEAD backend"; tail -40 "$BOOT_LOG"; cleanup; exit 1; }
    sleep 5
done

# Benchmark
log "Bench C4 (mix corpus)"
OUT=$OUTDIR/summ_${TAG}_C4_hostDSA_disabled_mix.json
PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
    --in "$SAMPLED" --method "C4_hostDSA_disabled" \
    --model "$MODEL" --model-tag "$TAG" \
    --port $PORT --max-tokens "$MAXTOK" \
    --concurrency "$CONC" --limit "$LIMIT" --shuffle \
    --out "$OUT" \
    > "$LOGD/C4_hostDSA_disabled_bench.log" 2>&1 \
    || log "  bench fail"

# Teardown
log "Teardown vllm"
pg=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
[ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
kill -9 "$PID" 2>/dev/null
PID=""
for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ -d "/proc/$op" ] && kill -9 "$op" 2>/dev/null
done
sleep 3

# Restore DSA
restore_dsa
log "Post-restore state"
for wq in "${WQ_LIST[@]}"; do
    s=$(cat /sys/bus/dsa/devices/$wq/state 2>/dev/null)
    log "  $wq: $s"
done

# Result
if [ -s "$OUT" ]; then
    tps=$($PY -c "import json; print(json.load(open('$OUT'))['output_tps'])" 2>/dev/null)
    gpu=$($PY -c "import json; print(json.load(open('$OUT'))['gpu_util'])" 2>/dev/null)
    log "=== C4_hostDSA_disabled tps=$tps gpu_util=$gpu ==="
fi
log "=== verify_host_dsa complete ==="
trap - INT TERM EXIT
