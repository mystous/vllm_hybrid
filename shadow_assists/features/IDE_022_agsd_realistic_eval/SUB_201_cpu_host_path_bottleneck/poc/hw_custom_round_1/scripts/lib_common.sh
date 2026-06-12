#!/usr/bin/env bash
# Common helpers for hw_custom_round_1 sweeps.
# IMPORTANT: never use `pkill -f vllm serve` (self-kill). Always use PID/PGID kill.

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_1
RUNS=$ROOT/runs
LOGS=$ROOT/logs
mkdir -p "$RUNS" "$LOGS"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8087
TP=8
MAX_MODEL_LEN=16384
CONC=64
NPROMPT=500
MAX_TOKENS=2048
SAMPLED=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_1/sharegpt500.parquet
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py

VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib

wait_gpu_free() {
    for i in $(seq 1 90); do
        local busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
        [ "$busy" -eq 0 ] && return 0
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
        [ -d "/proc/$pid" ] && kill -KILL -- -"$pid" 2>/dev/null || true
    fi
    for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
        [ -d "/proc/$op" ] && kill -KILL "$op" 2>/dev/null || true
    done
    sleep 3
    wait_gpu_free || true
}

wait_ready() {
    local boot_log_for_tag=$1
    for i in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        # Fast-fail: if boot log shows a fatal error, abort wait immediately.
        if [ -n "${boot_log_for_tag:-}" ] && [ -f "$boot_log_for_tag" ]; then
            if grep -qE "vllm serve: error|Engine core initialization failed|Engine startup failed|Failed core proc|^RuntimeError:|core dumped|Aborted \(core dumped\)|set_mempolicy.*Operation not permitted" "$boot_log_for_tag"; then
                echo "[wait_ready] fatal error detected in boot log; aborting" | tee -a "$boot_log_for_tag"
                return 1
            fi
        fi
        sleep 2
    done
    return 1
}

# Caller-set arrays (must be set per case via `declare -a EXTRA_ENV EXTRA_CLI`):
#   EXTRA_ENV  — list of "KEY=VAL" strings (added to env)
#   EXTRA_CLI  — list of CLI flag tokens (added at end of vllm serve)
#
# start_one TAG
start_one() {
    local tag=$1
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag env=[${EXTRA_ENV[*]:-}] cli=[${EXTRA_CLI[*]:-}]" | tee "$boot_log"

    local args=(
        env
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        VLLM_PREFETCH_TOKENIZE=1
        VLLM_BURST_AWARE_ADMISSION=1
        VLLM_WORKER_MULTIPROC_METHOD=spawn
    )
    if [ ${#EXTRA_ENV[@]} -gt 0 ]; then
        args+=( "${EXTRA_ENV[@]}" )
    fi
    args+=(
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        --allow-deprecated-quantization
    )
    # Only inject default compilation-config if caller hasn't supplied one
    local has_cc=0
    for tok in "${EXTRA_CLI[@]:-}"; do
        case "$tok" in
            --compilation-config|-cc) has_cc=1; break;;
        esac
    done
    if [ "$has_cc" -eq 0 ]; then
        args+=( --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' )
    fi
    if [ ${#EXTRA_CLI[@]} -gt 0 ]; then
        args+=( "${EXTRA_CLI[@]}" )
    fi
    echo "[boot] argc=${#args[@]}" >> "$boot_log"
    printf '[boot] arg: %s\n' "${args[@]}" >> "$boot_log"
    "${args[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    if ! wait_ready "$boot_log"; then
        echo "[boot] wait_ready timeout or fatal — pid=$pid" >> "$boot_log"
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

# bench_one TAG SWEEP_IDX
bench_one() {
    local tag=$1
    local sweep=${2:-1}
    "$VPY" "$RUNNER" --in "$SAMPLED" --method "hwc1_${tag}_s${sweep}" --model "$MODEL" --port "$PORT" \
        --max-tokens "$MAX_TOKENS" --concurrency "$CONC" --limit "$NPROMPT" \
        --shuffle --seed $((42 + sweep)) \
        --out "$RUNS/${tag}_s${sweep}.json" --raw "$RUNS/${tag}_s${sweep}.raw.jsonl" \
        2>&1 | tee -a "$LOGS/${tag}_bench.log"
}

stop_one() {
    local tag=$1
    local pid=$(cat $RUNS/${tag}.pid 2>/dev/null || echo "")
    [ -n "$pid" ] && kill_pgroup "$pid"
    rm -f $RUNS/${tag}.pid
}

# do_case_nsweep TAG N_SWEEPS
# Caller must `declare -a EXTRA_ENV EXTRA_CLI` first.
do_case_nsweep() {
    local tag=$1
    local n_sweeps=${2:-1}
    # If all sweeps already done, skip
    local all_done=1
    for s in $(seq 1 $n_sweeps); do
        [ -f "$RUNS/${tag}_s${s}.json" ] || all_done=0
    done
    if [ "$all_done" -eq 1 ]; then
        echo "[skip] $tag — all $n_sweeps sweeps already done"
        return 0
    fi
    if ! start_one "$tag"; then
        stop_one "$tag"
        echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > $RUNS/${tag}_s1.json
        return 1
    fi
    for s in $(seq 1 $n_sweeps); do
        if [ ! -f "$RUNS/${tag}_s${s}.json" ]; then
            bench_one "$tag" "$s" || true
        fi
        sleep 2
    done
    stop_one "$tag"
}
