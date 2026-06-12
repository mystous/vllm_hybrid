#!/usr/bin/env bash
# Eagle3 / Suffix final probe — last-chance lever discovery
# Baseline (이전 5 agent 확인): Llama-3.1-8B TP=8 vanilla = 22,008 tps (sharegpt 500p × conc=64 × max-tok=2048)
# Goal: +10% AND cpu↑
#
# 단계 1: Eagle3 (C-10a 0.7% accept 의심 → head 명시 + multi-conc)
# 단계 2: Suffix decoding (arctic SuffixDecodingCache, TSK_042 dominant lever 재현)
#
# 함정 (CLAUDE.md):
#   - pkill -f "vllm serve" 자기-매칭 금지 → PID/setsid pgroup kill
#   - --disable-log-requests 미지원
#   - orphan TP worker 는 nvidia-smi compute-apps PID 직접 kill
set -u

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/eagle3_suffix_final
RUNS=$ROOT/runs
MODEL="meta-llama/Llama-3.1-8B-Instruct"
EAGLE_HEAD="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
PORT=8005
TP=8
MAX_MODEL_LEN=16384
SAMPLED=$ROOT/sharegpt500.parquet
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py
VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm
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

# args: tag spec_json [extra_args ...]
start_one() {
    local tag=$1; shift
    local spec_json=$1; shift
    local boot_log=$RUNS/${tag}_boot.log
    echo "[boot] tag=$tag spec=$spec_json at $(date -u +%FT%TZ)" | tee "$boot_log"
    local cmd=(
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
    )
    if [ -n "$spec_json" ]; then
        cmd+=( --speculative-config "$spec_json" )
    fi
    cmd+=( "$@" )
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > "$RUNS/${tag}.pid"
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready 540; then
        echo "[boot] FAIL to come up" >> "$boot_log"
        tail -40 "$boot_log" >&2
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

# args: tag conc max_tokens nprompt
bench_one() {
    local tag=$1; shift
    local conc=$1; shift
    local max_tok=$1; shift
    local nprompt=$1; shift
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
        echo "[stop] tag=$tag pid=$pid at $(date -u +%FT%TZ)" >&2
        kill_pgroup "$pid"
    fi
    rm -f "$RUNS/${tag}.pid"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== Eagle3/Suffix final sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

PHASE="${PHASE:-all}"

# ===================== Phase 0 : baseline 재확인 =====================
# 사용자 baseline (이전 5 agent): 22,008 tps @ conc=64 × max-tok=2048
# Phase 0_brief 만 quick 확인 후 phase 1/2 진입.
if [ "$PHASE" = "all" ] || [ "$PHASE" = "0" ]; then
    if start_one "P0_baseline_c64" ""; then
        bench_one "P0_baseline_c64" 64 2048 500 || true
        stop_one  "P0_baseline_c64"
    fi
fi

# ===================== Phase 1 : Eagle3 multi-conc =====================
# Eagle3 head 명시 + num_spec_tokens=3 (vLLM 권장 기본)
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ]; then
    EAGLE_SPEC='{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":3}'
    if start_one "P1_E_boot" "$EAGLE_SPEC"; then
        bench_one "P1_E1_c8"   8  128  500 || true
        bench_one "P1_E2_c16"  16 512  500 || true
        bench_one "P1_E3_c32"  32 256  500 || true
        bench_one "P1_E4_c64"  64 2048 500 || true
        stop_one  "P1_E_boot"
    fi
fi

# ===================== Phase 2 : Suffix decoding =====================
# arctic_inference SuffixDecodingCache. 정상 동작이면 prefix repeat 잡음.
if [ "$PHASE" = "all" ] || [ "$PHASE" = "2" ]; then
    SUFFIX_SPEC='{"method":"suffix","num_speculative_tokens":8}'
    if start_one "P2_S_boot" "$SUFFIX_SPEC"; then
        bench_one "P2_S1_c8"   8  128  500 || true
        bench_one "P2_S2_c16"  16 256  500 || true
        bench_one "P2_S3_c32"  32 512  500 || true
        bench_one "P2_S4_c64"  64 2048 500 || true
        stop_one  "P2_S_boot"
    fi
fi

echo "==== sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
