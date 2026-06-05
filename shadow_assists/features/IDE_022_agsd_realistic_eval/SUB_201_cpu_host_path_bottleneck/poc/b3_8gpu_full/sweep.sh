#!/usr/bin/env bash
# B3 8GPU FULL 단독 강제 검증 sweep
# - GPU 0-7 (TP=8), Qwen2.5-7B-Instruct, port 8005
# - 모든 가용 backend × cudagraph_mode 시도 (focus: FULL 단독 활성 여부)
# - 측정: sharegpt 200p × conc=32 × max-tokens 8192 (stream → TTFT/TPOT)
#
# 함정 (CLAUDE.md):
#   - pkill -f "vllm serve" 자기-매칭 금지 → PID/setsid pgroup kill
#   - --disable-log-requests 미지원
#   - orphan TP worker 는 nvidia-smi compute-apps PID 직접 kill
#
# 시도 backend (B200 sm_100 호환):
#   - FLASHINFER (default)  — UNIFORM_BATCH cap
#   - FLASH_ATTN (FA4)      — UNIFORM_BATCH cap
#   - TRITON_ATTN           — ALWAYS cap   ★ FULL 단독 활성 후보
#   - FLEX_ATTENTION        — ALWAYS cap   ★ FULL 단독 활성 후보
#
# 효율 (총 boot 8회로 축소):
#   default FI : R0_FI_P (baseline) , R2_FI_FaP (current default)
#   FLASH_ATTN : R5_FA_FaP            (FA 강제 + FaP — 이전 4GPU R2_FA 와 비교용)
#   TRITON_ATTN: R6_TR_P, R7_TR_F, R8_TR_FaP   (3 mode 다 → ★ FULL 검증)
#   FLEX       : R10_FX_F                       (FULL 단독 활성 핵심)
#
# 추가로 R1_FI_F / R4_FA_F 는 이전 4GPU sweep 에서 FULL→FaP downgrade 가 확정됐으므로
# 8GPU 에서 boot log 만 채집 (no bench). do_boot_only 로 표시.
set -u

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full
RUNS=$ROOT/runs
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8005
TP=8
MAX_MODEL_LEN=16384
CONC=32
NPROMPT=200
MAX_TOKENS=8192
SAMPLED=$ROOT/sharegpt200.parquet
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
    for i in $(seq 1 540); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "[ready] port $PORT up after ${i}*2s" >&2
            return 0
        fi
        sleep 2
    done
    echo "[ready] TIMEOUT" >&2
    return 1
}

# args: tag attn_backend_json cudagraph_mode
start_one() {
    local tag=$1; shift
    local attn_json=$1; shift
    local cgmode=$1; shift
    local boot_log=$RUNS/${tag}_boot.log
    echo "[boot] tag=$tag attn=$attn_json cgmode=$cgmode at $(date -u +%FT%TZ)" | tee "$boot_log"
    local cmd=(
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        --compilation-config "{\"cudagraph_mode\":\"$cgmode\"}"
        --allow-deprecated-quantization
    )
    if [ -n "$attn_json" ]; then
        cmd+=( --attention-config "$attn_json" )
    fi
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready; then
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
        --method b3_8gpu \
        --model "$MODEL" \
        --port "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" \
        --limit "$NPROMPT" \
        --shuffle --seed 42 \
        --out "$summ" --raw "$raw" 2>&1 | tee -a "$RUNS/${tag}_bench.log"
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
    local tag=$1 attn=$2 cgmode=$3
    if ! start_one "$tag" "$attn" "$cgmode"; then
        echo "[case] $tag boot FAIL → skip bench"
        stop_one "$tag"
        return 1
    fi
    bench_one "$tag" || true
    stop_one "$tag"
}

do_boot_only() {
    local tag=$1 attn=$2 cgmode=$3
    if ! start_one "$tag" "$attn" "$cgmode"; then
        echo "[case-boot-only] $tag boot FAIL"
    fi
    stop_one "$tag"
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== B3 8GPU FULL sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

# Phase A : default FlashInfer 8GPU 재확인 (baseline + FaP)
do_case      "R0_FI_P"     ""                                  "PIECEWISE"
do_case      "R2_FI_FaP"   ""                                  "FULL_AND_PIECEWISE"
do_boot_only "R1_FI_F"     ""                                  "FULL"

# Phase B : FA forced
do_case      "R5_FA_FaP"   '{"backend":"FLASH_ATTN"}'          "FULL_AND_PIECEWISE"
do_boot_only "R4_FA_F"     '{"backend":"FLASH_ATTN"}'          "FULL"

# Phase C : TRITON_ATTN ★ ALWAYS cap → FULL 단독 활성 후보
do_case      "R6_TR_P"     '{"backend":"TRITON_ATTN"}'         "PIECEWISE"
do_case      "R7_TR_F"     '{"backend":"TRITON_ATTN"}'         "FULL"
do_case      "R8_TR_FaP"   '{"backend":"TRITON_ATTN"}'         "FULL_AND_PIECEWISE"

# Phase D : FLEX_ATTENTION ★ ALWAYS cap → FULL 단독 활성 후보
do_case      "R10_FX_F"    '{"backend":"FLEX_ATTENTION"}'      "FULL"

echo "==== sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
