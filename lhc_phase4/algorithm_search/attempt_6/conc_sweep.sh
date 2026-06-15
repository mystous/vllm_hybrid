#!/usr/bin/env bash
# Attempt 6 — discover the bottleneck by varying max-num-seqs.
# Hypothesis: vanilla baseline uses --max-num-seqs 64 with KV pool only
# 0.4% utilised. Raising concurrency without changing anything else
# should reveal whether the headroom is real (proves there's a lever).
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/algorithm_search/attempt_6
RUNS=${BASE}/runs/conc
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8516
TP=8
GPU_MEM=0.92
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"
INPUT_LEN=2304; OUTPUT_LEN=512; PREFIX_LEN=2048; NPROMPTS=500; MAX_LEN=4096

CONCS="${CONCS:-64 128 256}"
SWEEPS="${SWEEPS:-2}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

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
    sleep 3
}
trap cleanup EXIT

for CONC in $CONCS; do
    for SW in $(seq 1 $SWEEPS); do
        TAG="conc${CONC}_s${SW}"
        LOG="${RUNS}/${TAG}_boot.log"
        BENCH="${RUNS}/${TAG}_bench"
        [[ -s "${BENCH}.json" ]] && { echo "[$(ts)] skip $TAG"; continue; }

        echo "[$(ts)] === ${TAG} ===" | tee "${LOG}"

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
            echo "[$(ts)] not ready"
            cleanup
            continue
        fi
        echo "[$(ts)] ready"

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
            --save-result --result-dir "${RUNS}" \
            --result-filename "${TAG}_bench.json" \
            2>&1 | tee "${BENCH}.log"

        cleanup
    done
done

echo "[$(ts)] conc sweep done."
