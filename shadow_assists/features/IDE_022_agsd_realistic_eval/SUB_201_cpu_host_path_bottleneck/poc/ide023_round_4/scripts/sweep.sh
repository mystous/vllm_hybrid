#!/usr/bin/env bash
# IDE_023 Round 4 — IPC chunk + OINK ops + GPU mem util ↑ + max-num-seqs ↑↑
set -u
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_4
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

wait_gpu_free() { for i in $(seq 1 60); do local busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}'); [ "$busy" -eq 0 ] && return 0; sleep 2; done; return 1; }
kill_pgroup() { local pid=$1; [ -z "$pid" ] && return 0; if [ -d "/proc/$pid" ]; then kill -TERM -- -"$pid" 2>/dev/null || true; for i in $(seq 1 30); do [ -d "/proc/$pid" ] || break; sleep 1; done; [ -d "/proc/$pid" ] && kill -KILL -- -"$pid" 2>/dev/null || true; fi; for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do [ -d "/proc/$op" ] && kill -KILL "$op" 2>/dev/null || true; done; sleep 3; wait_gpu_free || true; }
wait_ready() { for i in $(seq 1 540); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

# Combined helper: takes tag, env_list, cli_list (separated by '|')
start_one() {
    local tag=$1; shift
    local extra_env="$1"; local extra_cli="$2"
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag env=[$extra_env] cli=[$extra_cli]" | tee "$boot_log"
    local cmd=( env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${BASE_ENV[@]}" )
    [ -n "$extra_env" ] && for e in $extra_env; do cmd+=( "$e" ); done
    cmd+=(
        setsid "$VLLM" serve "$MODEL"
        --tensor-parallel-size "$TP" --port "$PORT"
        --gpu-memory-utilization 0.85
        --max-model-len "$MAX_MODEL_LEN"
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
        --allow-deprecated-quantization
    )
    [ -n "$extra_cli" ] && for e in $extra_cli; do cmd+=( "$e" ); done
    echo "[boot] cmd: ${cmd[*]}" >> "$boot_log"
    "${cmd[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > $RUNS/${tag}.pid
    if ! wait_ready; then kill_pgroup "$pid"; return 1; fi
    return 0
}
bench_one() { local tag=$1; local summ=$RUNS/${tag}.json; local raw=$RUNS/${tag}.raw.jsonl; rm -f "$raw"; "$VPY" "$RUNNER" --in "$SAMPLED" --method "round4_$tag" --model "$MODEL" --port "$PORT" --max-tokens "$MAX_TOKENS" --concurrency "$CONC" --limit "$NPROMPT" --shuffle --seed 42 --out "$summ" --raw "$raw" 2>&1 | tee -a "$LOGS/${tag}_bench.log"; }
stop_one() { local tag=$1; local pid=$(cat $RUNS/${tag}.pid 2>/dev/null || echo ""); [ -n "$pid" ] && kill_pgroup "$pid"; rm -f $RUNS/${tag}.pid; }
do_case() { local tag=$1 env="$2" cli="$3"; [ -f $RUNS/${tag}.json ] && return 0; if ! start_one "$tag" "$env" "$cli"; then stop_one "$tag"; echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > $RUNS/${tag}.json; return 1; fi; bench_one "$tag" || true; stop_one "$tag"; }

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== Round 4 sweep start at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

do_case "r4a_mq_chunk64" "VLLM_MQ_MAX_CHUNK_BYTES_MB=64" ""
do_case "r4b_oink_ops" "VLLM_USE_OINK_OPS=1" ""
do_case "r4c_gpu_util95" "" "--gpu-memory-utilization 0.95"
do_case "r4d_maxseqs1024" "" "--max-num-seqs 1024 --max-num-batched-tokens 32768"

echo "==== Round 4 sweep complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
