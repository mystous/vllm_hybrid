#!/usr/bin/env bash
# SUB_201 후속 lever L1 — CPU AMX KV quantization measurement (BF16 → FP8)
#
# 목표: vLLM 의 --kv-cache-dtype 변경으로 KV memory ↓ → effective concurrency capacity ↑
#       memory-bound 회복 여부를 모델 크기별로 측정.
#
# 실제로 측정하는 것:
#   - baseline: --kv-cache-dtype auto  (= bf16/fp16, model dtype)
#   - fp8 KV : --kv-cache-dtype fp8    (B200 native = e4m3)
#
# 모델 별 conf:
#   M1 Qwen2.5-7B-Instruct      TP=2 GPU 0,1
#   M2 Llama-3.1-70B-Instruct   TP=4 GPU 0-3
#   M3 DeepSeek-R1 (671B)       TP=8 GPU 0-7   ★ boot ~5-7min, optional
#
# corpus: sharegpt 100p × conc=16 × max-tokens=512 (capacity-focused, short-output)
#
# 함정 (CLAUDE.md):
#   - LD_LIBRARY_PATH 명시 (torch/lib + nvidia/nccl/lib)
#   - pkill -f "vllm serve" 자기-매칭 금지 → PID/setsid pgroup kill
#   - --disable-log-requests 미지원
#   - orphan TP worker 는 nvidia-smi compute-apps PID 직접 kill
#   - 실제 모델명 전달 (completions payload)

set -u

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l1_kv_quant
RUNS=$ROOT/runs
SAMPLED=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full/sharegpt200.parquet
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py
VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

PORT=8013
CONC=16
NPROMPT=100
MAX_TOKENS=512
mkdir -p "$RUNS"

# LD_LIBRARY_PATH guard — 모든 vllm 명령 prefix
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib:/workspace/vllm_dev_prj/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

wait_gpu_free() {
    local gpus=$1
    local n_expect=$2
    for i in $(seq 1 60); do
        local busy
        busy=$(nvidia-smi -i "$gpus" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
        if [ "$busy" -eq 0 ]; then
            echo "[gpu] all $gpus free" >&2
            return 0
        fi
        sleep 2
    done
    echo "[gpu] WARN: timeout, still busy" >&2
    nvidia-smi -i "$gpus" --query-gpu=index,memory.used --format=csv,noheader >&2
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
    # orphan TP worker - nvidia-smi compute-apps PID 직접 kill
    local opids
    opids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u)
    for op in $opids; do
        if [ -d "/proc/$op" ]; then
            kill -KILL "$op" 2>/dev/null || true
        fi
    done
    sleep 2
}

wait_ready() {
    local timeout=$1
    for i in $(seq 1 "$timeout"); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "[ready] port $PORT up after ${i}*2s" >&2
            return 0
        fi
        sleep 2
    done
    echo "[ready] TIMEOUT after ${timeout}*2s" >&2
    return 1
}

snap_gpu_mem() {
    local gpus=$1
    local out=$2
    nvidia-smi -i "$gpus" --query-gpu=index,memory.used,memory.total --format=csv,noheader > "$out"
}

# args: tag model tp gpus ready_to kv_dtype maxlen
start_one() {
    local tag=$1; local model=$2; local tp=$3; local gpus=$4
    local ready_to=$5; local kv=$6; local maxlen=$7
    local boot_log=$RUNS/${tag}_boot.log
    echo "[boot] tag=$tag model=$model TP=$tp gpus=$gpus kv=$kv at $(date -u +%FT%TZ)" | tee "$boot_log"
    # b3_8gpu_full 에서 안정 검증된 옵션 + KV dtype 만 변경:
    #   FlashInfer default → PIECEWISE cudagraph (안정), enforce-eager 회피
    #   B11 finding: --enforce-eager + FlashInfer 조합에서 worker hang 재현 → 제외
    local cmd=(
        env CUDA_VISIBLE_DEVICES="$gpus"
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
        setsid "$VLLM" serve "$model"
        --tensor-parallel-size "$tp"
        --port "$PORT"
        --gpu-memory-utilization "${GMU:-0.85}"
        --max-model-len "$maxlen"
        --kv-cache-dtype "$kv"
        --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
    )
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    echo "[boot] pid=$pid" >> "$boot_log"
    if ! wait_ready "$ready_to"; then
        echo "[boot] FAIL to come up" >> "$boot_log"
        kill_pgroup "$pid"
        return 1
    fi
    # boot 직후 GPU 메모리 스냅 (KV cache 할당 ~포함된 정상화 상태)
    sleep 5
    snap_gpu_mem "$gpus" "$RUNS/${tag}_gpu_boot.csv"
    return 0
}

# args: tag model
bench_one() {
    local tag=$1; local model=$2
    local summ=$RUNS/${tag}.json
    local raw=$RUNS/${tag}.raw.jsonl
    local gpu_post=$RUNS/${tag}_gpu_post.csv
    rm -f "$raw"
    echo "[bench] $tag → $summ at $(date -u +%FT%TZ)" >&2
    "$VPY" "$RUNNER" \
        --in "$SAMPLED" \
        --method l1_kv_quant \
        --model "$model" \
        --port "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" \
        --limit "$NPROMPT" \
        --shuffle --seed 42 \
        --out "$summ" --raw "$raw" 2>&1 | tee -a "$RUNS/${tag}_bench.log"
    # bench 종료 후 GPU mem snapshot
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "$gpu_post"
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

# args: tag model tp gpus ready_to kv maxlen
do_case() {
    local tag=$1 model=$2 tp=$3 gpus=$4 ready_to=$5 kv=$6 maxlen=$7
    wait_gpu_free "$gpus" "$tp" || true
    if ! start_one "$tag" "$model" "$tp" "$gpus" "$ready_to" "$kv" "$maxlen"; then
        echo "[case] $tag boot FAIL → skip bench"
        stop_one "$tag"
        wait_gpu_free "$gpus" "$tp" || true
        return 1
    fi
    bench_one "$tag" "$model" || true
    stop_one "$tag"
    wait_gpu_free "$gpus" "$tp" || true
}

trap 'echo "[trap] interrupted"; exit 130' INT TERM
echo "==== L1 KV quant sweep start at $(date -u +%FT%TZ) ===="
echo "[env] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

PHASE=${PHASE:-all}

if [ "$PHASE" = "m1" ] || [ "$PHASE" = "all" ]; then
    # ---- M1: Qwen2.5-7B-Instruct, TP=2 GPU 0,1 ----
    do_case M1_qwen7b_auto  "Qwen/Qwen2.5-7B-Instruct" 2 "0,1" 180 "auto" 8192
    do_case M1_qwen7b_fp8   "Qwen/Qwen2.5-7B-Instruct" 2 "0,1" 180 "fp8"  8192
fi

if [ "$PHASE" = "m2" ] || [ "$PHASE" = "all" ]; then
    # ---- M2: Llama-3.1-70B-Instruct, TP=4 (외부 점유 회피 동적 선택) ----
    # M2_GPUS env 로 우회 (외부 작업 변동에 대응).
    do_case M2_llama70b_auto "meta-llama/Llama-3.1-70B-Instruct" 4 "${M2_GPUS:-0,1,4,5}" 360 "auto" 8192
    do_case M2_llama70b_fp8  "meta-llama/Llama-3.1-70B-Instruct" 4 "${M2_GPUS:-0,1,4,5}" 360 "fp8"  8192
fi

if [ "$PHASE" = "m2tp2" ]; then
    # M2 fallback: TP=2 (GPU 가용성이 4개 안 되는 환경에서)
    do_case M2t_llama70b_auto "meta-llama/Llama-3.1-70B-Instruct" 2 "${M2_GPUS:-0,1}" 360 "auto" 8192
    do_case M2t_llama70b_fp8  "meta-llama/Llama-3.1-70B-Instruct" 2 "${M2_GPUS:-0,1}" 360 "fp8"  8192
fi

if [ "$PHASE" = "m3" ] || [ "$PHASE" = "all" ]; then
    # ---- M3: DeepSeek-R1 (671B), TP=8 GPU 0-7 ----
    # R1 weight 자체가 native FP8 (sm_90 마이크로스케일 변형). baseline=auto 가 자동 fp8 weight,
    # KV-cache dtype lever 만 비교 (fp8 weight ≠ fp8 KV).
    do_case M3_r1_auto "deepseek-ai/DeepSeek-R1" 8 "0,1,2,3,4,5,6,7" 540 "auto" 8192
    do_case M3_r1_fp8  "deepseek-ai/DeepSeek-R1" 8 "0,1,2,3,4,5,6,7" 540 "fp8"  8192
fi

echo "==== sweep complete at $(date -u +%FT%TZ) ===="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
