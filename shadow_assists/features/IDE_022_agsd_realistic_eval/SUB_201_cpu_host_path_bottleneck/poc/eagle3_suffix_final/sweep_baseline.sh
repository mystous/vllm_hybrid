#!/usr/bin/env bash
# Vanilla baseline at matched concurrencies (for Eagle3 / Suffix Δ%).
# Reuses helpers (start_one / bench_one / stop_one / wait_gpu_free) from sweep.sh.
set -u

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/eagle3_suffix_final
RUNS=$ROOT/runs
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8005
TP=8
MAX_MODEL_LEN=16384
SAMPLED=$ROOT/sharegpt500.parquet
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py
VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

wait_gpu_free() {
    for i in $(seq 1 60); do
        local busy
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
        if [ "$busy" -eq 0 ]; then
            echo "[gpu] all 0-7 free" >&2
            return 0
        fi
        sleep 2
    done
    echo "[gpu] WARN timeout" >&2
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
    opids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u)
    for op in $opids; do
        if [ -d "/proc/$op" ]; then
            kill -KILL "$op" 2>/dev/null || true
        fi
    done
    sleep 2
    wait_gpu_free || true
}

wait_ready() {
    local timeout_iter=${1:-540}
    for i in $(seq 1 "$timeout_iter"); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "[ready] up after ${i}*2s" >&2
            return 0
        fi
        sleep 2
    done
    return 1
}

start_one() {
    local tag=$1
    local boot_log=$RUNS/${tag}_boot.log
    echo "[boot] $tag at $(date -u +%FT%TZ)" | tee "$boot_log"
    setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        "$VLLM" serve "$MODEL" \
        --tensor-parallel-size "$TP" \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MAX_MODEL_LEN" \
        --allow-deprecated-quantization \
        >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > "$RUNS/${tag}.pid"
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready 540; then
        echo "[boot] FAIL" >> "$boot_log"
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

bench_one() {
    local tag=$1; local conc=$2; local max_tok=$3; local nprompt=$4
    local summ=$RUNS/${tag}.json
    local raw=$RUNS/${tag}.raw.jsonl
    rm -f "$raw"
    echo "[bench] $tag conc=$conc max_tok=$max_tok n=$nprompt at $(date -u +%FT%TZ)" >&2
    "$VPY" "$RUNNER" \
        --in "$SAMPLED" \
        --method "$tag" \
        --model "$MODEL" \
        --port "$PORT" \
        --max-tokens "$max_tok" \
        --concurrency "$conc" \
        --limit "$nprompt" \
        --shuffle --seed 42 \
        --out "$summ" --raw "$raw" 2>&1 | tee -a "$RUNS/${tag}_bench.log"
}

stop_one() {
    local tag=$1
    local pid
    pid=$(cat "$RUNS/${tag}.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ]; then
        echo "[stop] $tag pid=$pid" >&2
        kill_pgroup "$pid"
    fi
    rm -f "$RUNS/${tag}.pid"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM

echo "==== vanilla baseline sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

# Single boot, multi-bench across (conc, max_tok) configs matching Eagle3/Suffix.
if start_one "BL_boot"; then
    bench_one "BL_c8_t128"   8  128  500 || true
    bench_one "BL_c16_t256"  16 256  500 || true
    bench_one "BL_c16_t512"  16 512  500 || true
    bench_one "BL_c32_t256"  32 256  500 || true
    bench_one "BL_c32_t512"  32 512  500 || true
    stop_one  "BL_boot"
fi

echo "==== complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
