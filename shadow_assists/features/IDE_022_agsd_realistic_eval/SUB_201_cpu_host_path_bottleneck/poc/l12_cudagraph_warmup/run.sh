#!/usr/bin/env bash
# L12 — predictive cudagraph warmup PoC sweep.
#
# 환경: B200 GPU 5 (only), Qwen2.5-7B-Instruct, TP=1.
# 시나리오: burst pattern (low → high concurrency 변환) — vanilla vs L12 hook.
#
# 함정 (CLAUDE.md / b3_8gpu_full):
#   - pkill -f "vllm serve" 자기-매칭 금지 → PID/setsid pgroup kill
#   - --disable-log-requests 미지원
#   - max-model-len 작게 (burst sample 은 짧은 prompt + max_tokens=512)
#   - orphan TP worker 직접 kill
set -u

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l12_cudagraph_warmup
RUNS=$ROOT/runs
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8112
TP=1
GPU=5
MAX_MODEL_LEN=4096          # 짧은 prompt + 512 max_tokens; 캡처 batch size 변동 더 잘 보임
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
    # Wait until GPU $GPU has < 500 MiB used AND no listed compute-apps
    # are running on its UUID (catches dying processes that still hold
    # memory).
    local gpu_uuid
    gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
        | awk -v g=$GPU -F', ' '$1==g {print $2}' | head -1)
    for i in $(seq 1 120); do
        local mb_used
        mb_used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -v g=$GPU -F', ' '$1==g {print $2}' | head -1)
        local apps
        apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
            | awk -v u="$gpu_uuid" -F', ' '$1==u {c++} END{print c+0}')
        if [ "${mb_used:-0}" -lt 500 ] && [ "$apps" -eq 0 ]; then
            return 0
        fi
        sleep 2
    done
    echo "[gpu] WARN: GPU $GPU still busy (mb=${mb_used} apps=${apps})" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null >&2
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
    # Orphan compute apps on our GPU (match on UUID, not GPU index)
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
            echo "[ready] port $PORT up after ${i}*2s" >&2
            return 0
        fi
        # Bail early if the parent serve PID is dead
        if [ "$expect_pid" -gt 0 ] && [ ! -d "/proc/$expect_pid" ]; then
            echo "[ready] FAIL — serve pid $expect_pid is gone after ${i}*2s" >&2
            return 2
        fi
        sleep 2
    done
    echo "[ready] TIMEOUT after $((180*2))s" >&2
    return 1
}

# args: tag predictive_mode (0|1|2)
start_one() {
    local tag=$1
    local mode=$2
    local boot_log=$RUNS/${tag}_boot.log

    echo "[boot] tag=$tag VLLM_CUDAGRAPH_PREDICTIVE_WARMUP=$mode at $(date -u +%FT%TZ)" | tee "$boot_log"

    # Lower log_every so the rank0 .jsonl has multiple records during the run
    env CUDA_VISIBLE_DEVICES=$GPU \
        VLLM_CUDAGRAPH_PREDICTIVE_WARMUP=$mode \
        VLLM_CUDAGRAPH_PREDICTIVE_LOG_EVERY=200 \
        VLLM_CUDAGRAPH_PREDICTIVE_LOG_PATH=$RUNS/${tag}_predictor.jsonl \
        setsid "$VLLM" serve "$MODEL" \
            --tensor-parallel-size "$TP" \
            --port "$PORT" \
            --gpu-memory-utilization 0.85 \
            --max-model-len "$MAX_MODEL_LEN" \
            >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready "$pid"; then
        echo "[boot] FAIL" >> "$boot_log"
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

bench_one() {
    local tag=$1
    local summ=$RUNS/${tag}.json
    echo "[bench] $tag → $summ at $(date -u +%FT%TZ)" >&2
    "$VPY" "$ROOT/burst_bench.py" \
        --in "$SAMPLED" \
        --model "$MODEL" \
        --port "$PORT" \
        --max-tokens "$MAX_TOK" \
        --warm-s "$WARM_S" --warm-rate "$WARM_RATE" \
        --ramp-s "$RAMP_S" \
        --hold-s "$HOLD_S" --peak-rate "$PEAK_RATE" \
        --cool-s "$COOL_S" \
        --out "$summ" 2>&1 | tee -a "$RUNS/${tag}_bench.log"
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
    local tag=$1 mode=$2 repeat=$3
    for r in $(seq 1 $repeat); do
        local rtag="${tag}_r${r}"
        local ok=0
        for attempt in 1 2; do
            if start_one "$rtag" "$mode"; then
                ok=1
                break
            fi
            echo "[case] $rtag boot attempt $attempt FAIL — wait+retry"
            stop_one "$rtag"
            sleep 10
        done
        if [ "$ok" -eq 0 ]; then
            echo "[case] $rtag boot FAIL after retries → skip bench"
            continue
        fi
        bench_one "$rtag" || true
        stop_one "$rtag"
    done
}

trap 'echo "[trap] interrupted"; stop_one current_running 2>/dev/null || true; exit 130' INT TERM
echo "==== L12 sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

REPEAT=${REPEAT:-2}
do_case "V0_vanilla"   "0" $REPEAT
wait_gpu_free || true
do_case "V1_observe"   "1" $REPEAT

echo "==== L12 sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
