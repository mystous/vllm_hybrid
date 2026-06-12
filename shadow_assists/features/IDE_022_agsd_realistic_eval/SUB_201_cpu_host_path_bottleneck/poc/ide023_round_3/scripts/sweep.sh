#!/usr/bin/env bash
# IDE_023 Round 3 — 새 env levers (output proc chunk, tokenize workers, flashinfer sampler, log stats interval)
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_3
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

BASE_ENV=( VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 )

wait_gpu_free() {
    for i in $(seq 1 60); do
        local busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
        [ "$busy" -eq 0 ] && { echo "[gpu] free" >&2; return 0; }
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
    sleep 3; wait_gpu_free || true
}
wait_ready() {
    for i in $(seq 1 540); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "[ready] up after ${i}*2s" >&2; return 0; }
        sleep 2
    done; return 1
}
# args: tag extra_env_list (space-sep KEY=VAL)
start_one() {
    local tag=$1; shift
    local extra_env="$1"
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag extra_env=[$extra_env] at $(date -u +%FT%TZ)" | tee "$boot_log"
    local cmd=( env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${BASE_ENV[@]}" )
    [ -n "$extra_env" ] && for e in $extra_env; do cmd+=( "$e" ); done
    cmd+=(
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
        --allow-deprecated-quantization
    )
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    if ! wait_ready; then kill_pgroup "$pid"; return 1; fi
    return 0
}
bench_one() {
    local tag=$1
    local summ=$RUNS/${tag}.json
    local raw=$RUNS/${tag}.raw.jsonl
    rm -f "$raw"
    echo "[bench] $tag at $(date -u +%FT%TZ)" >&2
    "$VPY" "$RUNNER" --in "$SAMPLED" --method "round3_$tag" \
        --model "$MODEL" --port "$PORT" --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" --limit "$NPROMPT" --shuffle --seed 42 \
        --out "$summ" --raw "$raw" 2>&1 | tee -a "$LOGS/${tag}_bench.log"
}
stop_one() {
    local tag=$1
    local pid=$(cat $RUNS/${tag}.pid 2>/dev/null || echo "")
    [ -n "$pid" ] && { echo "[stop] $tag pid=$pid" >&2; kill_pgroup "$pid"; }
    rm -f $RUNS/${tag}.pid
}
do_case() {
    local tag=$1 extra_env="$2"
    [ -f $RUNS/${tag}.json ] && { echo "[skip] $tag"; return 0; }
    if ! start_one "$tag" "$extra_env"; then
        stop_one "$tag"; echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > $RUNS/${tag}.json; return 1
    fi
    bench_one "$tag" || true
    stop_one "$tag"
}

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== Round 3 sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

do_case "r3a_outproc_chunk1024" "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=1024"
do_case "r3b_tok_workers16" "VLLM_PREFETCH_TOKENIZE_WORKERS=16"
do_case "r3c_flashinfer_sampler" "VLLM_USE_FLASHINFER_SAMPLER=1"
do_case "r3d_log_stats_100s" "VLLM_LOG_STATS_INTERVAL=100"

echo "==== Round 3 sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
