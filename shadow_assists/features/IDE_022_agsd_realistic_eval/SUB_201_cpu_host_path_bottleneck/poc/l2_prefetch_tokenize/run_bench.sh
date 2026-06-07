#!/usr/bin/env bash
# SUB_201 L2 — CPU prefetch + tokenization overlap
# B200 GPU 1만 사용. baseline vs prefetch_on 비교 측정.
#
# 사용:
#   bash run_bench.sh baseline
#   bash run_bench.sh prefetch_on
#   bash run_bench.sh both

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$HERE/runs"
mkdir -p "$RUNS_DIR"

PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=18221
HOST=127.0.0.1
GPU_INDEX=6
LOG_REQ="--no-enable-log-requests"

# Workload (proxy for sharegpt 200p × conc=16 × max-tok 256)
NUM_PROMPTS=200
CONCURRENCY=16
INPUT_LEN=512
OUTPUT_LEN=256
SEED=20260606
DATASET=random

export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib
export CUDA_VISIBLE_DEVICES=$GPU_INDEX
export VLLM_USE_V1=1

kill_pgroup() {
    local pid="$1"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        local pgid
        pgid=$(ps -o pgid= "$pid" | tr -d ' ' || true)
        if [[ -n "$pgid" ]]; then
            kill -TERM -- -"$pgid" 2>/dev/null || true
            sleep 2
            kill -KILL -- -"$pgid" 2>/dev/null || true
        fi
    fi
    # Also kill any orphan VLLM workers on our GPU
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    for p in $apps; do
        [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null || true
    done
    return 0
}

wait_ready() {
    local port="$1"
    local timeout="${2:-240}"
    local start=$(date +%s)
    while true; do
        if curl -sf "http://$HOST:$port/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        local now=$(date +%s)
        if (( now - start > timeout )); then
            echo "TIMEOUT waiting for server on port $port" >&2
            return 1
        fi
        sleep 2
    done
}

start_server() {
    local tag="$1"
    local log="$RUNS_DIR/server_${tag}.log"
    echo "[start] tag=$tag log=$log"
    setsid "$VLLM" serve "$MODEL" \
        --host "$HOST" --port "$PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.40 \
        --max-model-len 4096 \
        --max-num-seqs 64 \
        --enforce-eager \
        $LOG_REQ \
        >"$log" 2>&1 &
    SERVER_PID=$!
    echo "[start] server pid=$SERVER_PID"
    if ! wait_ready "$PORT" 300; then
        echo "[start] FAILED — tail of log:"
        tail -n 80 "$log" >&2 || true
        kill_pgroup "$SERVER_PID"
        return 1
    fi
    echo "[start] ready"
}

run_bench() {
    local tag="$1"
    local out_json="$RUNS_DIR/bench_${tag}.json"
    echo "[bench] tag=$tag out=$out_json"
    "$VLLM" bench serve \
        --backend openai \
        --model "$MODEL" \
        --base-url "http://$HOST:$PORT" \
        --endpoint /v1/completions \
        --dataset-name "$DATASET" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --seed "$SEED" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir "$RUNS_DIR" \
        --result-filename "bench_${tag}.json" \
        2>&1 | tee "$RUNS_DIR/bench_${tag}.stdout"
}

run_one() {
    local mode="$1"
    if [[ "$mode" == "baseline" ]]; then
        unset VLLM_PREFETCH_TOKENIZE
        unset VLLM_PREFETCH_TOKENIZE_WORKERS
    else
        export VLLM_PREFETCH_TOKENIZE=1
        export VLLM_PREFETCH_TOKENIZE_WORKERS=2
    fi
    SERVER_PID=""
    trap 'kill_pgroup "${SERVER_PID:-}"' EXIT
    start_server "$mode"
    sleep 5
    run_bench "$mode"
    kill_pgroup "${SERVER_PID:-}"
    trap - EXIT
    sleep 10
}

case "${1:-both}" in
    baseline) run_one baseline ;;
    prefetch_on) run_one prefetch_on ;;
    both) run_one baseline; run_one prefetch_on ;;
    *) echo "Usage: $0 [baseline|prefetch_on|both]"; exit 1 ;;
esac

echo "[done] results at $RUNS_DIR"
