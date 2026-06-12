#!/usr/bin/env bash
# Common library for hw_heavy_* rounds (H-1..H-5).
# Caller exports:
#   ROOT            absolute path of the round dir
#   MODEL           HF model id (or local path)
#   PORT            unique TCP port
#   TP              tensor-parallel size
#   MAX_MODEL_LEN
#   CONC            request concurrency for runner
#   NPROMPT
#   MAX_TOKENS
#   N_SWEEPS        number of repeat sweeps
#   EXTRA_ENV       array of `KEY=VAL` env tokens for the serve process
#   EXTRA_CLI       array of CLI tokens appended to `vllm serve ...`
#   EXTRA_LOG_TAG   logical tag prefix used in runner --method
# The serve process is started with `setsid` so a single PID = pgid.

RUNS=$ROOT/runs
LOGS=$ROOT/logs
mkdir -p "$RUNS" "$LOGS"

# shared dataset across rounds (already present from earlier rounds)
SAMPLED=${SAMPLED:-/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_1/sharegpt500.parquet}
RUNNER=${RUNNER:-/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py}
VPY=${VPY:-/workspace/vllm_dev_prj/bin/python}
VLLM=${VLLM:-/workspace/vllm_dev_prj/bin/vllm}
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
    local boot_log=$1
    for i in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        if [ -n "${boot_log:-}" ] && [ -f "$boot_log" ]; then
            if grep -qE "vllm serve: error|Engine core initialization failed|Engine startup failed|Failed core proc|^RuntimeError:|core dumped|Aborted \(core dumped\)|set_mempolicy.*Operation not permitted|ValueError:.*not support|AssertionError" "$boot_log"; then
                echo "[wait_ready] fatal" | tee -a "$boot_log"
                return 1
            fi
        fi
        sleep 2
    done
    return 1
}

start_one() {
    local tag=$1
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag model=$MODEL tp=$TP port=$PORT env=[${EXTRA_ENV[*]:-}] cli=[${EXTRA_CLI[*]:-}]" | tee "$boot_log"
    local args=( env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn )
    [ ${#EXTRA_ENV[@]} -gt 0 ] && args+=( "${EXTRA_ENV[@]}" )
    args+=( setsid "$VLLM" serve "$MODEL" \
        --tensor-parallel-size "$TP" \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MAX_MODEL_LEN" )
    [ ${#EXTRA_CLI[@]} -gt 0 ] && args+=( "${EXTRA_CLI[@]}" )
    printf '[boot] arg: %s\n' "${args[@]}" >> "$boot_log"
    "${args[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > "$RUNS/${tag}.pid"
    if ! wait_ready "$boot_log"; then
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

bench_one() {
    local tag=$1 sweep=${2:-1}
    "$VPY" "$RUNNER" \
        --in "$SAMPLED" \
        --method "${EXTRA_LOG_TAG:-heavy}_${tag}_s${sweep}" \
        --model "$MODEL" \
        --port "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" \
        --limit "$NPROMPT" \
        --shuffle --seed $((42 + sweep)) \
        --out "$RUNS/${tag}_s${sweep}.json" \
        --raw "$RUNS/${tag}_s${sweep}.raw.jsonl" \
        2>&1 | tee -a "$LOGS/${tag}_bench.log"
}

stop_one() {
    local tag=$1
    local pid=$(cat "$RUNS/${tag}.pid" 2>/dev/null || echo "")
    [ -n "$pid" ] && kill_pgroup "$pid"
    rm -f "$RUNS/${tag}.pid"
}

do_case_nsweep() {
    local tag=$1 n=${2:-$N_SWEEPS}
    local all_done=1
    for s in $(seq 1 $n); do [ -f "$RUNS/${tag}_s${s}.json" ] || all_done=0; done
    if [ "$all_done" -eq 1 ]; then echo "[skip] $tag"; return 0; fi
    if ! start_one "$tag"; then
        stop_one "$tag"
        echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > "$RUNS/${tag}_s1.json"
        return 1
    fi
    for s in $(seq 1 $n); do
        if [ ! -f "$RUNS/${tag}_s${s}.json" ]; then
            bench_one "$tag" "$s"
        fi
        sleep 2
    done
    stop_one "$tag"
}
