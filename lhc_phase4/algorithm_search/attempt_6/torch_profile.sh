#!/usr/bin/env bash
# Attempt 6 (v2) — torch.profiler via vllm /start_profile, /stop_profile.
# Captures a small window during steady-state for step-time decomposition.
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/algorithm_search/attempt_6
RUNS=${BASE}/runs
PROF=${RUNS}/torch_profile
mkdir -p "${PROF}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8515
TP=8
GPU_MEM=0.92
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"
# Use smaller window: 50 prompts to keep profile size manageable.
INPUT_LEN=2304; OUTPUT_LEN=128; PREFIX_LEN=2048; NPROMPTS=80; CONC=64; MAX_LEN=4096

LOG="${RUNS}/serve_profile_boot.log"
OWN_PIDS=()
cleanup() {
    for pid in "${OWN_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')
            [[ -n "$pgid" ]] && kill -- -"$pgid" 2>/dev/null || true
            sleep 2
            [[ -n "$pgid" ]] && kill -9 -- -"$pgid" 2>/dev/null || true
        fi
    done
    OWN_PIDS=()
}
trap cleanup EXIT

echo "[boot] starting vllm with profiler enabled"
VLLM_TORCH_PROFILER_DIR="${PROF}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    setsid /workspace/vllm_dev_prj/bin/vllm serve ${MODEL} \
      --port ${PORT} --host 127.0.0.1 \
      --tensor-parallel-size ${TP} \
      --gpu-memory-utilization ${GPU_MEM} \
      --max-model-len ${MAX_LEN} \
      --max-num-seqs ${CONC} \
      --enable-prefix-caching \
      >> "${LOG}" 2>&1 &
SERVE_PID=$!
OWN_PIDS+=("$SERVE_PID")

READY=0
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1; break
    fi
    sleep 5
done
[[ $READY -eq 0 ]] && { echo "[boot] not ready"; exit 1; }
echo "[boot] ready"

# Send a small warmup request so the engine is in steady state.
curl -s "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":4}" >/dev/null
sleep 2

# Now: kick off background benchmark + profile a 3s window during it.
(
    /workspace/vllm_dev_prj/bin/vllm bench serve \
        --model "${MODEL}" \
        --dataset-name sonnet \
        --dataset-path "${DATA}" \
        --sonnet-input-len ${INPUT_LEN} \
        --sonnet-output-len ${OUTPUT_LEN} \
        --sonnet-prefix-len ${PREFIX_LEN} \
        --num-prompts ${NPROMPTS} \
        --max-concurrency ${CONC} \
        --port ${PORT} \
        > "${RUNS}/bench.log" 2>&1
) &
BENCH_PID=$!
OWN_PIDS+=("$BENCH_PID")

# Wait until at least 64 requests are queued.
sleep 3

echo "[prof] starting torch profiler"
curl -s -X POST "http://127.0.0.1:${PORT}/start_profile"
sleep 4
echo "[prof] stopping torch profiler"
curl -s -X POST "http://127.0.0.1:${PORT}/stop_profile"
echo "[prof] profile capture done"

wait "$BENCH_PID" || true
sync
sleep 3
echo "[done] files:"
ls -la "${PROF}/" | head -20
