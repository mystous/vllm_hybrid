#!/usr/bin/env bash
# [SUB_201/L10] seed 99 only
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l10_admission
RUNS=$ROOT/runs

MODEL=Qwen/Qwen2.5-7B-Instruct
PORT=8016
GPU=4
TP=1
MAX_MODEL_LEN=20480
SAMPLED=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full/sharegpt200.parquet
N_REQ=400
BURST_MAX=32
BURST_DUR=0.05
IDLE_MEAN=0.6
SEED=99

VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

wait_gpu_free() {
    for i in $(seq 1 60); do
        local busy
        busy=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {print 1; exit}')
        if [ -z "$busy" ]; then return 0; fi
        sleep 2
    done
    return 1
}
kill_pgroup() {
    local pid=$1
    [ -z "$pid" ] && return 0
    if [ -d "/proc/$pid" ]; then
        kill -TERM -- -"$pid" 2>/dev/null || true
        for i in $(seq 1 30); do [ -d "/proc/$pid" ] || break; sleep 1; done
        if [ -d "/proc/$pid" ]; then kill -KILL -- -"$pid" 2>/dev/null || true; fi
    fi
}
wait_ready() {
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        if grep -q "Engine core initialization failed" $1 2>/dev/null; then return 1; fi
        sleep 2
    done
    return 1
}
start_one() {
    local tag=$1 burst=$2
    local boot=$RUNS/${tag}_boot99.log
    echo "[boot] $tag burst=$burst at $(date -u +%FT%TZ)" | tee "$boot"
    local env_args=(env CUDA_VISIBLE_DEVICES=$GPU)
    [ "$burst" = "1" ] && env_args+=(VLLM_BURST_AWARE_ADMISSION=1)
    "${env_args[@]}" setsid "$VLLM" serve "$MODEL" \
        --tensor-parallel-size "$TP" \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MAX_MODEL_LEN" \
        --allow-deprecated-quantization \
        >> "$boot" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid99
    if ! wait_ready "$boot"; then
        kill_pgroup "$pid"; return 1
    fi
    return 0
}
bench_one() {
    local tag=$1
    local summ=$RUNS/${tag}_s${SEED}.json
    local raw=$RUNS/${tag}_s${SEED}.raw.jsonl
    echo "[bench] $tag seed=$SEED → $summ" >&2
    "$VPY" "$ROOT/burst_client.py" \
        --in "$SAMPLED" --model "$MODEL" --port "$PORT" \
        --n-requests "$N_REQ" --burst-max "$BURST_MAX" \
        --burst-dur-s "$BURST_DUR" --idle-mean-s "$IDLE_MEAN" \
        --tag "${tag}_s${SEED}" --out "$summ" --raw "$raw" \
        --seed "$SEED" 2>&1 | tee -a "$RUNS/${tag}_s${SEED}_bench.log"
}
stop_one() {
    local pid
    pid=$(cat $RUNS/${1}.pid99 2>/dev/null || echo "")
    [ -n "$pid" ] && kill_pgroup "$pid"
    rm -f $RUNS/${1}.pid99
    wait_gpu_free || true
}
do_case() {
    for attempt in 1 2 3; do
        if start_one "$1" "$2"; then break; fi
        echo "[case] $1 boot attempt $attempt FAIL"
        stop_one "$1"; sleep 5
    done
    [ ! -f "$RUNS/${1}.pid99" ] && { echo "[case] $1 all attempts FAIL"; return 1; }
    bench_one "$1" || true
    stop_one "$1"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== L10 seed99 sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
do_case BASELINE 0
do_case BURSTAWARE 1
echo "==== L10 seed99 sweep complete at $(date -u +%FT%TZ) ===="
