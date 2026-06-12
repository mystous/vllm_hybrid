#!/usr/bin/env bash
# IDE_023 Round 1 — 워크로드 확대 baseline + 새 lever 4종
#  - Model: meta-llama/Llama-3.1-8B-Instruct, TP=8
#  - Workload: sharegpt 500p × conc=64 × max-tok=2048 (host overhead 가시화)
#  - Baseline = Optimal Config (vanilla + FaP + L2 + L10)
#  - 새 lever (vLLM-native, 미시도):
#      R1A — stream-interval=8 (host send overhead 감소)
#      R1B — max-num-batched-tokens=16384 (default ~8K → 16K)
#      R1C — max-num-seqs=512 (default 256 → 512)
#      R1D — long-prefill-token-threshold=512 + max-num-partial-prefills=4 (Sarathi-Serve)
#
# 시간 박스: round 4-6시간
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_1
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
SAMPLED=$ROOT/sharegpt500.parquet
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py

VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib

# Optimal Config (B3 FaP + L2 + L10)
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
    local boot_to=${1:-540}
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

# args: tag extra_cli_args (space-sep CLI flags appended to vllm serve)
start_one() {
    local tag=$1; shift
    local extra_cli="$1"
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag extra_cli=[$extra_cli] at $(date -u +%FT%TZ)" | tee "$boot_log"
    local cmd=(
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        "${BASE_ENV[@]}"
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
        --allow-deprecated-quantization
    )
    if [ -n "$extra_cli" ]; then
        # Split by whitespace into individual cli args
        for e in $extra_cli; do
            cmd+=( "$e" )
        done
    fi
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
        --method "round1_$tag" \
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
    local tag=$1 extra_cli="$2"
    if [ -f $RUNS/${tag}.json ]; then
        echo "[skip] $tag already has summary"
        return 0
    fi
    if ! start_one "$tag" "$extra_cli"; then
        echo "[case] $tag boot FAIL → skip bench"
        stop_one "$tag"
        echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > $RUNS/${tag}.json
        return 1
    fi
    bench_one "$tag" || true
    stop_one "$tag"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== IDE_023 Round 1 sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

# Baseline (Optimal Config, no extra cli) — 워크로드 확대
do_case "baseline" ""

# R1A: stream-interval=8 (host send overhead 감소)
do_case "r1a_stream_interval8" "--stream-interval 8"

# R1B: max-num-batched-tokens=16384 (larger batch chunk)
do_case "r1b_batch16k" "--max-num-batched-tokens 16384"

# R1C: max-num-seqs=512 (double default)
do_case "r1c_maxseqs512" "--max-num-seqs 512"

# R1D: Sarathi-Serve — long-prefill-token-threshold + partial prefills
do_case "r1d_sarathi" "--long-prefill-token-threshold 512 --max-num-partial-prefills 4 --max-long-partial-prefills 2"

echo "==== Round 1 sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
