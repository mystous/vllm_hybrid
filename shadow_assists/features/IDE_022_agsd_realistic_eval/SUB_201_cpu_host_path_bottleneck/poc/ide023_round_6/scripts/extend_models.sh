#!/usr/bin/env bash
# IDE_023 Round 6 — R6A KV-cache-dtype=fp8 모델 확장 측정
#   Llama-3.1-70B TP=8 + Qwen-2.5-32B TP=8 + DeepSeek-R1-Distill-Llama-70B TP=8
#   각 모델: baseline (no fp8) vs R6A (fp8) 측정 → Δ% 비교
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_6
RUNS=$ROOT/runs
LOGS=$ROOT/logs
mkdir -p "$RUNS" "$LOGS"

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

wait_gpu_free() { for i in $(seq 1 60); do local busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}'); [ "$busy" -eq 0 ] && return 0; sleep 2; done; return 1; }
kill_pgroup() { local pid=$1; [ -z "$pid" ] && return 0; if [ -d "/proc/$pid" ]; then kill -TERM -- -"$pid" 2>/dev/null || true; for i in $(seq 1 30); do [ -d "/proc/$pid" ] || break; sleep 1; done; [ -d "/proc/$pid" ] && kill -KILL -- -"$pid" 2>/dev/null || true; fi; for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do [ -d "/proc/$op" ] && kill -KILL "$op" 2>/dev/null || true; done; sleep 3; wait_gpu_free || true; }
wait_ready() { for i in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

# args: tag model cgmode kv_dtype
start_one() {
    local tag=$1; local model=$2; local cgmode=$3; local kv_dtype=$4
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag model=$model cg=$cgmode kv=$kv_dtype" | tee "$boot_log"
    local args=(
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        "${BASE_ENV[@]}"
        setsid "$VLLM" serve "$model"
        --tensor-parallel-size "$TP"
        --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        --compilation-config "{\"cudagraph_mode\":\"$cgmode\"}"
        --allow-deprecated-quantization
    )
    [ -n "$kv_dtype" ] && args+=( --kv-cache-dtype "$kv_dtype" )
    echo "[boot] cmd: ${args[*]}" >> "$boot_log"
    "${args[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    if ! wait_ready; then kill_pgroup "$pid"; return 1; fi
    return 0
}
bench_one() { local tag=$1; local model=$2; "$VPY" "$RUNNER" --in "$SAMPLED" --method "ext_$tag" --model "$model" --port "$PORT" --max-tokens "$MAX_TOKENS" --concurrency "$CONC" --limit "$NPROMPT" --shuffle --seed 42 --out "$RUNS/${tag}.json" --raw "$RUNS/${tag}.raw.jsonl" 2>&1 | tee -a "$LOGS/${tag}_bench.log"; }
stop_one() { local tag=$1; local pid=$(cat $RUNS/${tag}.pid 2>/dev/null || echo ""); [ -n "$pid" ] && kill_pgroup "$pid"; rm -f $RUNS/${tag}.pid; }
do_case() { local tag=$1 model=$2 cg=$3 kv=$4; [ -f $RUNS/${tag}.json ] && return 0; if ! start_one "$tag" "$model" "$cg" "$kv"; then stop_one "$tag"; echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > $RUNS/${tag}.json; return 1; fi; bench_one "$tag" "$model" || true; stop_one "$tag"; }

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== R6 model extension start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

# 70B DeepSeek-R1-Distill-Llama (cached) — baseline + R6A
do_case "ext_ds70b_base"  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" "FULL_AND_PIECEWISE" ""
do_case "ext_ds70b_fp8"   "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" "FULL_AND_PIECEWISE" "fp8"

# 32B Qwen 2.5 — baseline + R6A
do_case "ext_q32b_base"  "Qwen/Qwen2.5-32B-Instruct" "FULL_AND_PIECEWISE" ""
do_case "ext_q32b_fp8"   "Qwen/Qwen2.5-32B-Instruct" "FULL_AND_PIECEWISE" "fp8"

echo "==== R6 model extension complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
