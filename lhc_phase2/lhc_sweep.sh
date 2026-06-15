#!/usr/bin/env bash
# LHC Phase 2 — DSA lane integration sweep (2 workload × 3 config × 3 sweep).
#
# Configs:
#   vanilla         — no LHC, no FaP, no fp8 KV
#   lhc_dsa         — VLLM_LHC_DSA=1 + VLLM_LEVER_N9=1 (DSA lane active in KV
#                     staging fan-out; gated by env so it falls through to
#                     ATen scatter when device/lib not available)
#   lhc_dsa_fp8kv   — LHC DSA + fp8 KV cache (best stack at 100+ lever review)
#
# Workloads:
#   sonnet          — sharegpt 500p prefix-repeat workload
#   balanced        — mixed-len realistic distribution
#
# Methodology mirrors six_workload_sweep/sweep.sh — per-method boot 1x,
# per-workload bench, full pgroup kill, GPU mem 0 verify, then next.
set -u

ROOT=/workspace/host_vllm_hybrid/lhc_phase2
RUNS=$ROOT/runs_sweep
MODEL="meta-llama/Llama-3.1-8B-Instruct"
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
WORKLOADS=(sonnet balanced)
SWEEPS=3

mkdir -p "$RUNS"

wait_gpu_free() {
    for i in $(seq 1 60); do
        local busy
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '$1>500 {c++} END{print c+0}')
        if [ "$busy" -eq 0 ]; then
            echo "[gpu] all 0-7 free" >&2
            return 0
        fi
        sleep 2
    done
    echo "[gpu] WARN timeout" >&2
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

start_one() {
    local tag=$1; shift
    local cfg=$1; shift
    local boot_log=$RUNS/${tag}_boot.log
    echo "[boot] tag=$tag cfg=$cfg at $(date -u +%FT%TZ)" | tee "$boot_log"
    local extra_env=()
    local extra_cli=()
    case "$cfg" in
        vanilla)
            ;;
        lhc_dsa)
            extra_env=(VLLM_LHC_DSA=1 VLLM_LEVER_N9=1 VLLM_LHC_DSA_MIN=65536)
            ;;
        lhc_dsa_fp8kv)
            extra_env=(VLLM_LHC_DSA=1 VLLM_LEVER_N9=1 VLLM_LHC_DSA_MIN=65536)
            extra_cli=(--kv-cache-dtype fp8)
            ;;
        *)
            echo "[boot] unknown cfg=$cfg" >&2; return 1 ;;
    esac

    local cmd=(
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${extra_env[@]}"
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        "${extra_cli[@]}"
    )
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > "$RUNS/${tag}.pid"
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready 540; then
        echo "[boot] FAIL" >> "$boot_log"
        tail -120 "$boot_log" >&2
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

bench_one() {
    local cfg=$1; shift
    local wl=$1; shift
    local sw=$1; shift
    local tag="${cfg}_${wl}_s${sw}"
    local out=$RUNS/${tag}.json
    local log=$RUNS/${tag}_bench.log
    echo "[bench] $tag n=$NPROMPTS max_tok=$MAX_TOKENS conc=$CONC at $(date -u +%FT%TZ)" >&2
    "$VPY" "$HARNESS" \
        --scenario vanilla \
        --workload "$wl" \
        --num-prompts "$NPROMPTS" \
        --target-input-len "$TARGET_IN" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" \
        --model "$MODEL" \
        --sonnet "$SONNET" \
        --seed $((42 + sw)) \
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
echo "==== LHC Phase 2 sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

CONFIGS_DEFAULT=(vanilla lhc_dsa lhc_dsa_fp8kv)
CONFIGS_ARG="${CONFIGS:-${CONFIGS_DEFAULT[*]}}"
read -r -a CONFIGS_LIST <<< "$CONFIGS_ARG"

for cfg in "${CONFIGS_LIST[@]}"; do
    boot_tag="boot_${cfg}"
    if start_one "$boot_tag" "$cfg"; then
        for wl in "${WORKLOADS[@]}"; do
            for sw in $(seq 1 $SWEEPS); do
                bench_one "$cfg" "$wl" "$sw" || true
            done
        done
        stop_one "$boot_tag"
    else
        echo "[main] $cfg boot FAILED — skipping" >&2
    fi
    wait_gpu_free || true
done

echo "==== sweep complete at $(date -u +%FT%TZ) ===="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
