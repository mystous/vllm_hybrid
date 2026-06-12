#!/usr/bin/env bash
# Six-workload sweep — sonnet/chat/code/balanced/sonnet-heavy/code-heavy
# Methods: vanilla / eagle3 / suffix / fp8_kv
# Purpose: prefix-repeat 다른 corpus 에서 spec decode 및 fp8 KV 의 net effect 확인.
#
# Design:
#   - method 별 server boot 1 회 → 6 workload 연속 bench → kill (boot/kill 4 회로 제한)
#   - 500 prompts × max-tokens=2048 × conc=64 × target_input_len=1024 (boot/bench 시간 합리적)
#   - benchmark_workloads.py 의 scenario=vanilla (port 8001) 사용
#
# Safety:
#   - PID/setsid pgroup kill (자기-pkill 금지)
#   - orphan TP worker → nvidia-smi compute-apps PID kill
#   - GPU 0 MiB 확인 후 다음 method boot
set -u

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/six_workload_sweep
RUNS=$ROOT/runs
MODEL="meta-llama/Llama-3.1-8B-Instruct"
EAGLE_HEAD="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
PORT=8001
TP=8
MAX_MODEL_LEN=16384
SONNET=/workspace/host_vllm_hybrid/benchmarks/sonnet.txt
HARNESS=/workspace/host_vllm_hybrid/vllm_config_perf/gating/benchmark_workloads.py
VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm
NPROMPTS=500
MAX_TOKENS=2048
TARGET_IN=1024
CONC=64
WORKLOADS=(sonnet chat code balanced sonnet-heavy code-heavy)

mkdir -p "$RUNS"

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
    sleep 2
    wait_gpu_free || true
}

wait_ready() {
    local timeout_iter=${1:-540}
    for i in $(seq 1 "$timeout_iter"); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "[ready] port $PORT up after ${i}*2s" >&2
            return 0
        fi
        sleep 2
    done
    echo "[ready] TIMEOUT" >&2
    return 1
}

# args: tag method
start_one() {
    local tag=$1; shift
    local method=$1; shift
    local boot_log=$RUNS/${tag}_boot.log
    echo "[boot] tag=$tag method=$method at $(date -u +%FT%TZ)" | tee "$boot_log"

    local cmd=(
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
    )
    case "$method" in
        vanilla)
            ;;
        eagle3)
            cmd+=( --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":3}' )
            ;;
        suffix)
            cmd+=( --speculative-config '{"method":"suffix","num_speculative_tokens":8}' )
            ;;
        fp8_kv)
            cmd+=( --kv-cache-dtype fp8 )
            ;;
        *)
            echo "[boot] unknown method=$method" >&2
            return 1
            ;;
    esac
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > "$RUNS/${tag}.pid"
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready 540; then
        echo "[boot] FAIL" >> "$boot_log"
        tail -60 "$boot_log" >&2
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

# args: method workload [nprompt] [max_tok]
bench_one() {
    local method=$1; shift
    local workload=$1; shift
    local nprompt=${1:-$NPROMPTS}
    local max_tok=${2:-$MAX_TOKENS}
    local tag="${method}_${workload}"
    local out=$RUNS/${tag}.json
    local log=$RUNS/${tag}_bench.log
    echo "[bench] $tag n=$nprompt max_tok=$max_tok conc=$CONC at $(date -u +%FT%TZ)" >&2
    "$VPY" "$HARNESS" \
        --scenario vanilla \
        --workload "$workload" \
        --num-prompts "$nprompt" \
        --target-input-len "$TARGET_IN" \
        --max-tokens "$max_tok" \
        --concurrency "$CONC" \
        --model "$MODEL" \
        --sonnet "$SONNET" \
        --seed 42 \
        --out "$out" 2>&1 | tee "$log"
}

stop_one() {
    local tag=$1
    local pid
    pid=$(cat "$RUNS/${tag}.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ]; then
        echo "[stop] tag=$tag pid=$pid at $(date -u +%FT%TZ)" >&2
        kill_pgroup "$pid"
    fi
    rm -f "$RUNS/${tag}.pid"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== six-workload sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

METHODS_DEFAULT=(vanilla eagle3 suffix fp8_kv)
METHODS_ARG="${METHODS:-${METHODS_DEFAULT[*]}}"
read -r -a METHODS_LIST <<< "$METHODS_ARG"

for method in "${METHODS_LIST[@]}"; do
    boot_tag="boot_${method}"
    if start_one "$boot_tag" "$method"; then
        for wl in "${WORKLOADS[@]}"; do
            bench_one "$method" "$wl" || true
        done
        stop_one "$boot_tag"
    else
        echo "[main] $method boot FAILED — skipping" >&2
    fi
    wait_gpu_free || true
done

echo "==== sweep complete at $(date -u +%FT%TZ) ===="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
