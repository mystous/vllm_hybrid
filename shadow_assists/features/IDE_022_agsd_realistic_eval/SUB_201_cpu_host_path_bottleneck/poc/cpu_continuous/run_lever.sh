#!/usr/bin/env bash
# cpu_continuous — single-lever 5-sweep runner
# usage:
#   run_lever.sh <tag> <extra_env> <extra_cli>
# example:
#   run_lever.sh c7a "VLLM_FOO=1" "--kv-cache-dtype fp8 --calculate-kv-scales"
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/cpu_continuous
RUNS=$ROOT/runs
LOGS=$ROOT/runs
mkdir -p "$RUNS"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8093
TP=8
MAX_MODEL_LEN=16384
CONC=64
NPROMPT=500
MAX_TOKENS=2048
SAMPLED=$ROOT/sharegpt500.parquet
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py
VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib

BASE_ENV=(
    VLLM_PREFETCH_TOKENIZE=1
    VLLM_BURST_AWARE_ADMISSION=1
)

TAG=$1; shift
EXTRA_ENV=$1; shift
EXTRA_CLI=$1; shift
NSWEEP=${1:-5}

wait_gpu_free() {
    for i in $(seq 1 60); do
        local busy
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
        if [ "$busy" -eq 0 ]; then return 0; fi
        sleep 2
    done
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
        if [ -d "/proc/$pid" ]; then kill -KILL -- -"$pid" 2>/dev/null || true; fi
    fi
    local opids
    opids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u)
    for op in $opids; do
        if [ -d "/proc/$op" ]; then kill -KILL "$op" 2>/dev/null || true; fi
    done
    sleep 3
    wait_gpu_free || true
}

wait_ready() {
    for i in $(seq 1 540); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

boot_log=$RUNS/${TAG}_boot.log
echo "[boot] tag=$TAG env=[$EXTRA_ENV] cli=[$EXTRA_CLI] at $(date -u +%FT%TZ)" | tee "$boot_log"

# Build env array (BASE + EXTRA)
env_args=( "${BASE_ENV[@]}" )
if [ -n "$EXTRA_ENV" ]; then
    for kv in $EXTRA_ENV; do
        env_args+=( "$kv" )
    done
fi

cmd=(
    env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    "${env_args[@]}"
    setsid "$VLLM" serve "$MODEL"
    --tensor-parallel-size "$TP"
    --port "$PORT"
    --gpu-memory-utilization 0.85
    --max-model-len "$MAX_MODEL_LEN"
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --allow-deprecated-quantization
)
if [ -n "$EXTRA_CLI" ]; then
    for e in $EXTRA_CLI; do
        cmd+=( "$e" )
    done
fi
echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
"${cmd[@]}" >> "$boot_log" 2>&1 &
PID=$!
echo "$PID" > $RUNS/${TAG}.pid
echo "[boot] pid=$PID" >> "$boot_log"

trap 'echo "[trap] interrupted"; kill_pgroup $PID; exit 130' INT TERM

if ! wait_ready; then
    echo "[boot] FAIL" | tee -a "$boot_log"
    kill_pgroup $PID
    rm -f $RUNS/${TAG}.pid
    echo "{\"tag\":\"$TAG\",\"status\":\"boot_fail\"}" > $RUNS/${TAG}.json
    exit 1
fi

# 5-sweep
for s in $(seq 1 $NSWEEP); do
    summ=$RUNS/${TAG}_s${s}.json
    raw=$RUNS/${TAG}_s${s}.raw.jsonl
    rm -f "$raw"
    echo "[bench] $TAG s$s → $summ at $(date -u +%FT%TZ)"
    "$VPY" "$RUNNER" \
        --in "$SAMPLED" \
        --method "cpu_cont_$TAG" \
        --model "$MODEL" \
        --port "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" \
        --limit "$NPROMPT" \
        --shuffle --seed $((42+s)) \
        --out "$summ" --raw "$raw" 2>&1 | tail -3
done

kill_pgroup $PID
rm -f $RUNS/${TAG}.pid
echo "[done] $TAG sweep complete"
