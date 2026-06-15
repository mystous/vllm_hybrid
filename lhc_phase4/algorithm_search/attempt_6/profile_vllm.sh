#!/usr/bin/env bash
# Attempt 6 — py-spy attach during steady-state load to find host gap.
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/algorithm_search/attempt_6
RUNS=${BASE}/runs
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8514
TP=8
GPU_MEM=0.92
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"
INPUT_LEN=2304; OUTPUT_LEN=512; PREFIX_LEN=2048; NPROMPTS=500; CONC=64; MAX_LEN=4096

LOG="${RUNS}/serve_boot.log"
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

echo "[boot] Starting vllm serve..."
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
if [[ $READY -eq 0 ]]; then
    echo "[boot] vllm not ready"; exit 1
fi
echo "[boot] vllm ready"

# Find the engine core PID for py-spy attach.
ENGINE_PIDS=$(pgrep -f "VLLM::EngineCore" | head -5)
if [[ -z "$ENGINE_PIDS" ]]; then
    # Fallback — find any python child of SERVE_PID's process tree
    ENGINE_PIDS=$(pgrep -P "$SERVE_PID" -f "python" | head -5)
fi
echo "[boot] engine PIDs: $ENGINE_PIDS" | tee -a "${LOG}"

# Run benchmark for 30s while py-spy samples on the first engine core.
FIRST_PID=$(echo "$ENGINE_PIDS" | head -1)
if [[ -z "$FIRST_PID" ]]; then
    echo "[boot] no engine PID found"
    pgrep -af "vllm" | head -20
    exit 1
fi
echo "[boot] sampling engine PID $FIRST_PID"

# Background py-spy
/workspace/vllm_dev_prj/bin/py-spy record \
    --pid "$FIRST_PID" \
    --duration 30 \
    --output "${RUNS}/engine_profile.svg" \
    --format flamegraph \
    --rate 100 \
    --idle \
    > "${RUNS}/pyspy.log" 2>&1 &
PYSPY_PID=$!
OWN_PIDS+=("$PYSPY_PID")

# Concurrently kick off the benchmark.
sleep 2
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

# Also collect a per-function dump using py-spy dump.
/workspace/vllm_dev_prj/bin/py-spy dump --pid "$FIRST_PID" > "${RUNS}/py_dump.txt" 2>&1 || true

# Wait for py-spy to finish.
wait "$PYSPY_PID" 2>/dev/null || true

echo "[done] profile saved to ${RUNS}/engine_profile.svg"
ls -la "${RUNS}/"
