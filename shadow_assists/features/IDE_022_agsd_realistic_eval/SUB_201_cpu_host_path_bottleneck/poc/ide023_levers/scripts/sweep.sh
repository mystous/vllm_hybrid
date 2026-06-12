#!/usr/bin/env bash
# IDE_023 13-lever sequential PoC sweep
#  - Model: meta-llama/Llama-3.1-8B-Instruct (32 heads → TP=8 OK)
#  - GPU 0-7 (TP=8), port 8087
#  - Baseline = Optimal Config (vanilla + FaP + L2 + L10), measured ONCE
#  - Each lever: ON-only re-boot + bench (compared against shared baseline)
#  - Bench: sharegpt 200p × conc=16 × max-tok 512
#
# Each lever box: 6 hour cap (boot ≤8 min, bench ≤25 min)
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_levers
RUNS=$ROOT/runs
LOGS=$ROOT/logs
mkdir -p "$RUNS" "$LOGS"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8087
TP=8
MAX_MODEL_LEN=16384
CONC=16
NPROMPT=200
MAX_TOKENS=512
SAMPLED=$ROOT/sharegpt200.parquet
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py

VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib

# Baseline env (Optimal Config: B3 FaP + L2 + L10)
BASE_ENV=(
    VLLM_PREFETCH_TOKENIZE=1
    VLLM_BURST_AWARE_ADMISSION=1
)

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
    echo "[gpu] WARN: timeout, still busy" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
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
    sleep 3
    wait_gpu_free || true
}

wait_ready() {
    local boot_to=${1:-540}  # 18 min default
    for i in $(seq 1 $boot_to); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "[ready] port $PORT up after ${i}*2s" >&2
            return 0
        fi
        sleep 2
    done
    echo "[ready] TIMEOUT" >&2
    return 1
}

# args: tag extra_env_string (space-sep, e.g. "VLLM_LEVER_N1=1")
start_one() {
    local tag=$1; shift
    local extra_env_str="$1"
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag extra_env=[$extra_env_str] at $(date -u +%FT%TZ)" | tee "$boot_log"
    # build the env command prefix
    local cmd=(
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        "${BASE_ENV[@]}"
    )
    if [ -n "$extra_env_str" ]; then
        # Split by whitespace into env entries
        for e in $extra_env_str; do
            cmd+=( "$e" )
        done
    fi
    cmd+=(
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}"
        --allow-deprecated-quantization
    )
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready 540; then
        echo "[boot] FAIL to come up" >> "$boot_log"
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

bench_one() {
    local tag=$1
    local summ=$RUNS/${tag}.json
    local raw=$RUNS/${tag}.raw.jsonl
    rm -f "$raw"
    echo "[bench] $tag → $summ at $(date -u +%FT%TZ)" >&2
    "$VPY" "$RUNNER" \
        --in "$SAMPLED" \
        --method ide023 \
        --model "$MODEL" \
        --port "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" \
        --limit "$NPROMPT" \
        --shuffle --seed 42 \
        --out "$summ" --raw "$raw" 2>&1 | tee -a "$LOGS/${tag}_bench.log"
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

do_case() {
    local tag=$1 envs="$2"
    if [ -f $RUNS/${tag}.json ]; then
        echo "[skip] $tag already has summary"
        return 0
    fi
    if ! start_one "$tag" "$envs"; then
        echo "[case] $tag boot FAIL → skip bench"
        stop_one "$tag"
        # Mark as fail
        echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > $RUNS/${tag}.json
        return 1
    fi
    bench_one "$tag" || true
    stop_one "$tag"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== IDE_023 lever sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

# Baseline (Optimal Config, no lever)
do_case "baseline" ""

# 13 levers, one at a time
for L in N1 N4 N5 N6 N7 N8 N9 N10 N11 N14 N17 N19 N20; do
    do_case "lever_${L}" "VLLM_LEVER_${L}=1"
done

echo "==== sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
