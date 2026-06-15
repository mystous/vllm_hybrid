#!/usr/bin/env bash
# Attempt 6 verification — does the conc lever hold across workloads?
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/algorithm_search/attempt_6
RUNS=${BASE}/runs/multi_wl
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8517
TP=8
GPU_MEM=0.92
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"
MAX_LEN=4096

WORKLOADS="${WORKLOADS:-sonnet chat_short chat_prefix}"
CONCS="${CONCS:-64 128 256}"
SWEEPS="${SWEEPS:-3}"

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

for WORKLOAD in $WORKLOADS; do
    case "$WORKLOAD" in
        sonnet)      INPUT_LEN=512;  OUTPUT_LEN=512; PREFIX_LEN=0;    NPROMPTS=500 ;;
        chat_short)  INPUT_LEN=512;  OUTPUT_LEN=128; PREFIX_LEN=0;    NPROMPTS=500 ;;
        chat_prefix) INPUT_LEN=2304; OUTPUT_LEN=512; PREFIX_LEN=2048; NPROMPTS=500 ;;
    esac
    for CONC in $CONCS; do
        for SW in $(seq 1 $SWEEPS); do
            TAG="${WORKLOAD}_c${CONC}_s${SW}"
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
                echo "[$(ts)] not ready"; cleanup; continue
            fi

            # Warmup ping to mitigate s1 cold-start penalty.
            curl -s "http://127.0.0.1:${PORT}/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hi\",\"max_tokens\":4}" >/dev/null
            sleep 1

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
                2>&1 | tee "${BENCH}.log" | grep -E "Total token|Mean TPOT|Mean TTFT|req/s"

            cleanup
        done
    done
done

echo "[$(ts)] multi-wl sweep done"
