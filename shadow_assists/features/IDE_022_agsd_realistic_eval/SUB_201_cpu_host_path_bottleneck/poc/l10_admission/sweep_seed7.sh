#!/usr/bin/env bash
# [SUB_201/L10] 2nd-seed = 7 only (variance evidence, fewer reboots)
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l10_admission
RUNS=$ROOT/runs
mkdir -p "$RUNS"

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
SEED=7

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
        # detect engine init failure early
        if grep -q "Engine core initialization failed" $1 2>/dev/null; then
            echo "[ready] engine init failed (early-exit)" >&2
            return 1
        fi
        sleep 2
    done
    return 1
}

start_one() {
    local tag=$1 burst=$2
    local boot=$RUNS/${tag}_boot.log
    echo "[boot] $tag burst=$burst at $(date -u +%FT%TZ)" | tee "$boot"
    local env_args=(env CUDA_VISIBLE_DEVICES=$GPU)
    if [ "$burst" = "1" ]; then
        env_args+=(VLLM_BURST_AWARE_ADMISSION=1)
    fi
    "${env_args[@]}" setsid "$VLLM" serve "$MODEL" \
        --tensor-parallel-size "$TP" \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MAX_MODEL_LEN" \
        --allow-deprecated-quantization \
        >> "$boot" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    if ! wait_ready "$boot"; then
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

bench_one() {
    local tag=$1 seed=$2
    local summ=$RUNS/${tag}_s${seed}.json
    local raw=$RUNS/${tag}_s${seed}.raw.jsonl
    echo "[bench] $tag seed=$seed → $summ at $(date -u +%FT%TZ)" >&2
    "$VPY" "$ROOT/burst_client.py" \
        --in "$SAMPLED" \
        --model "$MODEL" \
        --port "$PORT" \
        --n-requests "$N_REQ" \
        --burst-max "$BURST_MAX" \
        --burst-dur-s "$BURST_DUR" \
        --idle-mean-s "$IDLE_MEAN" \
        --tag "${tag}_s${seed}" \
        --out "$summ" --raw "$raw" \
        --seed "$seed" 2>&1 | tee -a "$RUNS/${tag}_s${seed}_bench.log"
}

stop_one() {
    local tag=$1
    local pid
    pid=$(cat $RUNS/${tag}.pid 2>/dev/null || echo "")
    if [ -n "$pid" ]; then kill_pgroup "$pid"; fi
    rm -f $RUNS/${tag}.pid
    wait_gpu_free || true
}

do_case() {
    local tag=$1 burst=$2
    for attempt in 1 2 3; do
        if start_one "$tag" "$burst"; then break; fi
        echo "[case] $tag boot attempt $attempt FAIL → retry"
        stop_one "$tag"
        sleep 5
    done
    if [ ! -f "$RUNS/${tag}.pid" ]; then
        echo "[case] $tag all attempts FAIL — skipping bench"
        return 1
    fi
    bench_one "$tag" "$SEED" || true
    stop_one "$tag"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== L10 seed7 sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

do_case BASELINE 0
do_case BURSTAWARE 1

echo "==== L10 seed7 sweep complete at $(date -u +%FT%TZ) ===="
