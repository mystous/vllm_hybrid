#!/usr/bin/env bash
# Alternating V0/V1 sweep to isolate hook effect from host state.
# Sequence: V0_a r1, V1_a r1, V0_a r2, V1_a r2, V0_a r3, V1_a r3
# Result tags differ from run.sh (V0_alt/V1_alt) so aggregate can keep
# the original sweep2 isolated.
set -u

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l12_cudagraph_warmup
RUNS=$ROOT/runs
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8112
GPU=5
MAX_MODEL_LEN=4096
SAMPLED=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full/sharegpt200.parquet
VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm
mkdir -p "$RUNS"

# Burst pattern config
WARM_S=15
WARM_RATE=2
RAMP_S=12
HOLD_S=25
PEAK_RATE=18
COOL_S=8
MAX_TOK=384

wait_gpu_free() {
    local gpu_uuid
    gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
        | awk -v g=$GPU -F', ' '$1==g {print $2}' | head -1)
    for i in $(seq 1 120); do
        local mb_used apps
        mb_used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -v g=$GPU -F', ' '$1==g {print $2}' | head -1)
        apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
            | awk -v u="$gpu_uuid" -F', ' '$1==u {c++} END{print c+0}')
        if [ "${mb_used:-0}" -lt 500 ] && [ "$apps" -eq 0 ]; then
            return 0
        fi
        sleep 2
    done
    echo "[gpu] WARN: GPU $GPU still busy" >&2
    return 1
}

kill_pgroup() {
    local pid=$1
    [ -z "$pid" ] && return 0
    local gpu_uuid
    gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
        | awk -v g=$GPU -F', ' '$1==g {print $2}' | head -1)
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
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
        | awk -v u="$gpu_uuid" -F', ' '$1==u {print $2}' \
        | while read op; do
            [ -n "$op" ] && [ -d "/proc/$op" ] && kill -KILL "$op" 2>/dev/null || true
          done
    sleep 3
    wait_gpu_free || true
}

wait_ready() {
    local expect_pid=${1:-0}
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        if [ "$expect_pid" -gt 0 ] && [ ! -d "/proc/$expect_pid" ]; then
            echo "[ready] FAIL — serve pid $expect_pid is gone after ${i}*2s" >&2
            return 2
        fi
        sleep 2
    done
    echo "[ready] TIMEOUT" >&2
    return 1
}

start_one() {
    local tag=$1
    local mode=$2
    local boot_log=$RUNS/${tag}_boot.log
    echo "[boot] tag=$tag mode=$mode at $(date -u +%FT%TZ)" | tee "$boot_log"
    env CUDA_VISIBLE_DEVICES=$GPU \
        VLLM_CUDAGRAPH_PREDICTIVE_WARMUP=$mode \
        VLLM_CUDAGRAPH_PREDICTIVE_LOG_EVERY=200 \
        VLLM_CUDAGRAPH_PREDICTIVE_LOG_PATH=$RUNS/${tag}_predictor.jsonl \
        setsid "$VLLM" serve "$MODEL" \
            --tensor-parallel-size 1 \
            --port "$PORT" \
            --gpu-memory-utilization 0.85 \
            --max-model-len "$MAX_MODEL_LEN" \
            >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    if ! wait_ready "$pid"; then
        echo "[boot] FAIL" >> "$boot_log"
        kill_pgroup "$pid"
        return 1
    fi
    echo "[boot] $tag READY" >&2
    return 0
}

bench_one() {
    local tag=$1
    local summ=$RUNS/${tag}.json
    "$VPY" "$ROOT/burst_bench.py" \
        --in "$SAMPLED" --model "$MODEL" --port "$PORT" \
        --max-tokens "$MAX_TOK" \
        --warm-s "$WARM_S" --warm-rate "$WARM_RATE" \
        --ramp-s "$RAMP_S" --hold-s "$HOLD_S" --peak-rate "$PEAK_RATE" \
        --cool-s "$COOL_S" \
        --out "$summ" 2>&1 | tee "$RUNS/${tag}_bench.log"
}

stop_one() {
    local tag=$1
    local pid
    pid=$(cat $RUNS/${tag}.pid 2>/dev/null || echo "")
    if [ -n "$pid" ]; then
        echo "[stop] tag=$tag pid=$pid at $(date -u +%FT%TZ)" >&2
        kill_pgroup "$pid"
    fi
    rm -f $RUNS/${tag}.pid
}

do_one() {
    local tag=$1 mode=$2
    for attempt in 1 2; do
        if start_one "$tag" "$mode"; then
            bench_one "$tag" || true
            stop_one "$tag"
            return 0
        fi
        echo "[case] $tag boot attempt $attempt FAIL"
        stop_one "$tag"
        sleep 10
    done
    return 1
}

echo "==== L12 alt sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

# Alternate: V0, V1, V0, V1, V0, V1
do_one "V0_alt_r1" "0"
do_one "V1_alt_r1" "1"
do_one "V0_alt_r2" "0"
do_one "V1_alt_r2" "1"
do_one "V0_alt_r3" "0"
do_one "V1_alt_r3" "1"

echo "==== L12 alt sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
