#!/usr/bin/env bash
# [SUB_201/L10] burst-aware admission sweep
#  - Qwen2.5-7B-Instruct, TP=1, GPU index 4 only
#  - port 8016
#  - baseline (FCFS) vs burst-aware (VLLM_BURST_AWARE_ADMISSION=1)
#  - sharegpt prompt + bimodal max_tokens + Poisson bursts
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

VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

wait_gpu_free() {
    for i in $(seq 1 90); do
        local busy
        busy=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {print 1; exit}')
        if [ -z "$busy" ]; then
            echo "[gpu] GPU $GPU free" >&2
            return 0
        fi
        sleep 2
    done
    echo "[gpu] WARN: timeout, still busy" >&2
    return 1
}

kill_pgroup() {
    local pid=$1
    [ -z "$pid" ] && return 0
    if [ -d "/proc/$pid" ]; then
        kill -TERM -- -"$pid" 2>/dev/null || true
        for i in $(seq 1 30); do
            [ -d "/proc/$pid" ] || break
            sleep 1
        done
        if [ -d "/proc/$pid" ]; then
            kill -KILL -- -"$pid" 2>/dev/null || true
        fi
    fi
    local opids
    opids=$(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader | sort -u)
    for op in $opids; do
        if [ -d "/proc/$op" ]; then
            kill -KILL "$op" 2>/dev/null || true
        fi
    done
    sleep 2
}

wait_ready() {
    for i in $(seq 1 540); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "[ready] port $PORT up after ${i}*2s" >&2
            return 0
        fi
        sleep 2
    done
    echo "[ready] TIMEOUT" >&2
    return 1
}

# args: tag burst_flag(0|1)
start_one() {
    local tag=$1
    local burst=$2
    local boot=$RUNS/${tag}_boot.log
    echo "[boot] $tag burst=$burst at $(date -u +%FT%TZ)" | tee "$boot"
    local env_args=(env CUDA_VISIBLE_DEVICES=$GPU)
    if [ "$burst" = "1" ]; then
        env_args+=(VLLM_BURST_AWARE_ADMISSION=1)
        env_args+=(VLLM_BURST_TRIGGER_DEPTH=4)
        env_args+=(VLLM_BURST_HEAD_WINDOW=16)
        env_args+=(VLLM_BURST_AGE_CAP_S=2.0)
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
    echo "[boot] pid=$pid" >> "$boot"
    if ! wait_ready; then
        echo "[boot] FAIL" >> "$boot"
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

bench_one() {
    local tag=$1
    local summ=$RUNS/${tag}.json
    local raw=$RUNS/${tag}.raw.jsonl
    echo "[bench] $tag → $summ at $(date -u +%FT%TZ)" >&2
    "$VPY" "$ROOT/burst_client.py" \
        --in "$SAMPLED" \
        --model "$MODEL" \
        --port "$PORT" \
        --n-requests "$N_REQ" \
        --burst-max "$BURST_MAX" \
        --burst-dur-s "$BURST_DUR" \
        --idle-mean-s "$IDLE_MEAN" \
        --tag "$tag" \
        --out "$summ" --raw "$raw" \
        --seed 42 2>&1 | tee -a "$RUNS/${tag}_bench.log"
}

stop_one() {
    local tag=$1
    local pid
    pid=$(cat $RUNS/${tag}.pid 2>/dev/null || echo "")
    if [ -n "$pid" ]; then
        echo "[stop] $tag pid=$pid" >&2
        kill_pgroup "$pid"
    fi
    rm -f $RUNS/${tag}.pid
    wait_gpu_free || true
}

do_case() {
    local tag=$1 burst=$2
    if ! start_one "$tag" "$burst"; then
        echo "[case] $tag boot FAIL → skip"
        stop_one "$tag"
        return 1
    fi
    bench_one "$tag" || true
    stop_one "$tag"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== L10 admission sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

do_case BASELINE 0
do_case BURSTAWARE 1

echo "==== L10 sweep complete at $(date -u +%FT%TZ) ===="
nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader
