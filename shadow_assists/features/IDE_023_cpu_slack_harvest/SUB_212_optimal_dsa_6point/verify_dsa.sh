#!/usr/bin/env bash
# Verify the +36% vanilla gain source: DSA (사용자 가설) vs vllm code drift.
#
# Llama-3.1-8B mix corpus, vanilla config (no DSA env), n=1.
# 4 cells (C0 already measured):
#   C0: current vllm + LHC dir (already = 12,089)
#   C1 = "B": current vllm + LHC dir REMOVED
#   C2 = "A": old vllm (2026-05-27 commit a203ecd5e) + LHC dir intact
#   C3 = "A+B": old vllm + LHC dir REMOVED
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

OLD_COMMIT="a203ecd5e"
LHC_DIR="/workspace/host_vllm_hybrid/vllm/v1/lhc"
LHC_BAK="/tmp/lhc_dir_bak_$$"

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
PID=""
LHC_MOVED=0
GIT_STASHED=0
GIT_OLD_HEAD=""

restore_all(){
    log "[restore] PID=$PID LHC_MOVED=$LHC_MOVED GIT_STASHED=$GIT_STASHED"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        log "  kill vllm pid=$PID"
        local pg; pg=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
        [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
        kill -9 "$PID" 2>/dev/null
        for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
            [ -d "/proc/$op" ] && kill -9 "$op" 2>/dev/null
        done
        PID=""
    fi
    if [ -n "$GIT_OLD_HEAD" ]; then
        log "  git checkout back to $GIT_OLD_HEAD"
        git checkout "$GIT_OLD_HEAD" 2>&1 | tail -3
        GIT_OLD_HEAD=""
    fi
    if [ "$GIT_STASHED" -eq 1 ]; then
        log "  git stash pop"
        git stash pop 2>&1 | tail -3
        GIT_STASHED=0
    fi
    if [ "$LHC_MOVED" -eq 1 ]; then
        log "  restore LHC dir from $LHC_BAK"
        mv "$LHC_BAK" "$LHC_DIR" || true
        LHC_MOVED=0
    fi
}
trap 'log "[trap] interrupted"; restore_all; exit 130' INT TERM EXIT

wait_gpu_free(){
    for i in $(seq 1 60); do
        local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s+0}')
        [ "${u:-1}" -lt 4000 ] && { log "  gpu freed"; return 0; }
        sleep 5
    done
    log "  WARN gpu not freed"; return 0
}

wait_ready(){
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "  READY"; return 0; }
        [ -n "${1:-}" ] && ! kill -0 "$1" 2>/dev/null && { log "  DEAD backend"; return 1; }
        sleep 5
    done
    log "  TIMEOUT"; return 1
}

run_cell(){
    local cell_tag=$1
    local out=$OUTDIR/summ_${TAG}_${cell_tag}_mix.json
    local boot_log=$LOGD/${cell_tag}_boot.log
    : > "$boot_log"
    log "  boot vllm ($cell_tag)"

    env CUDA_VISIBLE_DEVICES=$GPUS setsid "$VBIN" serve "$MODEL" \
        --tensor-parallel-size $TP --port $PORT \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MML" \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        > "$boot_log" 2>&1 < /dev/null &
    PID=$!
    if ! wait_ready "$PID"; then
        log "  $cell_tag boot FAILED"
        tail -50 "$boot_log"
        kill -9 "$PID" 2>/dev/null || true
        PID=""
        return 1
    fi

    log "  bench $cell_tag (mix corpus)"
    PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
        --in "$SAMPLED" --method "$cell_tag" \
        --model "$MODEL" --model-tag "$TAG" \
        --port $PORT --max-tokens "$MAXTOK" \
        --concurrency "$CONC" --limit "$LIMIT" --shuffle \
        --out "$out" \
        >> "$LOGD/${cell_tag}_bench.log" 2>&1 \
        || log "  bench fail"

    # tear down
    local pg; pg=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
    kill -9 "$PID" 2>/dev/null
    for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
        [ -d "/proc/$op" ] && kill -9 "$op" 2>/dev/null
    done
    PID=""
    wait_gpu_free
    if [ -s "$out" ]; then
        local tps; tps=$($PY -c "import json; print(json.load(open('$out'))['output_tps'])" 2>/dev/null)
        log "  $cell_tag tps=$tps"
    else
        log "  $cell_tag NO OUTPUT"
    fi
}

log "=== verify_dsa start ==="
wait_gpu_free

# ─── C1: current vllm + LHC removed ─────────────────────────────────
log "########## C1 = B: current vllm + LHC REMOVED ##########"
if [ -d "$LHC_DIR" ]; then
    mv "$LHC_DIR" "$LHC_BAK"
    LHC_MOVED=1
    log "  moved $LHC_DIR -> $LHC_BAK"
fi
run_cell "C1_curvllm_noLHC"
# Restore LHC
mv "$LHC_BAK" "$LHC_DIR"
LHC_MOVED=0
log "  restored LHC dir"

# ─── C2: old vllm + LHC intact ──────────────────────────────────────
log "########## C2 = A: old vllm ($OLD_COMMIT) + LHC INTACT ##########"
GIT_OLD_HEAD=$(git rev-parse HEAD)
log "  current HEAD=$GIT_OLD_HEAD"
git stash push -m "verify_dsa_temp" -- vllm/ 2>&1 | tail -3
GIT_STASHED=1
git checkout "$OLD_COMMIT" 2>&1 | tail -3
run_cell "C2_oldvllm_withLHC"
# Restore git
git checkout "$GIT_OLD_HEAD" 2>&1 | tail -3
GIT_OLD_HEAD=""
git stash pop 2>&1 | tail -3
GIT_STASHED=0

# ─── C3: old vllm + LHC removed ─────────────────────────────────────
log "########## C3 = A+B: old vllm + LHC REMOVED ##########"
GIT_OLD_HEAD=$(git rev-parse HEAD)
git stash push -m "verify_dsa_temp" -- vllm/ 2>&1 | tail -3
GIT_STASHED=1
git checkout "$OLD_COMMIT" 2>&1 | tail -3
if [ -d "$LHC_DIR" ]; then
    mv "$LHC_DIR" "$LHC_BAK"
    LHC_MOVED=1
fi
run_cell "C3_oldvllm_noLHC"
if [ "$LHC_MOVED" -eq 1 ]; then
    mv "$LHC_BAK" "$LHC_DIR"
    LHC_MOVED=0
fi
git checkout "$GIT_OLD_HEAD" 2>&1 | tail -3
GIT_OLD_HEAD=""
git stash pop 2>&1 | tail -3
GIT_STASHED=0

log "=== verify_dsa complete ==="
trap - INT TERM EXIT
echo "RESULTS:"
for cell in C1_curvllm_noLHC C2_oldvllm_withLHC C3_oldvllm_noLHC; do
    out=$OUTDIR/summ_${TAG}_${cell}_mix.json
    if [ -s "$out" ]; then
        $PY -c "import json; d=json.load(open('$out')); print(f'  $cell: tps={d[\"output_tps\"]} gpu={d[\"gpu_util\"]} ttft={d[\"ttft_ms_p50\"]}')"
    else
        echo "  $cell: MISSING"
    fi
done
echo "  C0 baseline (current vllm + LHC) = 12,089 (already measured)"
