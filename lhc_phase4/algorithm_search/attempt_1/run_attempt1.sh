#!/usr/bin/env bash
# LHC Phase 4 / Attempt 1 — libamx_c3.so production build measurement.
#
# Hypothesis: With libamx_c3.so (C-FNV-1a) replacing the python FNV-1a
# loop, the Path 1 -2.53% regression converges toward 0 or becomes
# positive. Microbench: ctypes-FNV 1.94 us/call vs SHA256 0.76 us/call.
# (Negative microbench suggests gate FAIL, but we measure end-to-end
# anyway because batching/scheduler interactions may differ.)
#
# 1 sweep, chat_prefix only (binding workload from Path 1).

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/algorithm_search/attempt_1
RUNS=${BASE}/runs
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8513
TP=8
GPU_MEM=0.92
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"

CONFIGS="${CONFIGS:-vanilla lhc_amx_c3_clib}"
SWEEPS="${SWEEPS:-1}"
# chat_prefix: 2048 prefix + 256 user + 512 output -> ~88% prefix-cache hit
INPUT_LEN=2304; OUTPUT_LEN=512; PREFIX_LEN=2048; NPROMPTS=500; CONC=64; MAX_LEN=4096
WORKLOAD=chat_prefix

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

OWN_PIDS=()
cleanup_my_pids() {
    for pid in "${OWN_PIDS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')
            if [[ -n "$pgid" ]]; then
                kill -- -"$pgid" 2>/dev/null || true
                sleep 2
                kill -9 -- -"$pgid" 2>/dev/null || true
            fi
        fi
    done
    OWN_PIDS=()
    sleep 4
}
trap cleanup_my_pids EXIT

for CONFIG in $CONFIGS; do
    case "$CONFIG" in
        vanilla)
            ENV_PRE=""
            ;;
        lhc_amx_c3_clib)
            ENV_PRE="VLLM_LHC_AMX_C3_PREFIX=1 VLLM_LHC_AMX_C3_LIB=/workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so"
            ;;
    esac

    for SWEEP in $(seq 1 $SWEEPS); do
        TAG="a1_${WORKLOAD}_${CONFIG}_s${SWEEP}"
        LOG="${RUNS}/${TAG}_boot.log"
        BENCH_PREFIX="${RUNS}/${TAG}_bench"

        if [[ -s "${BENCH_PREFIX}.json" ]]; then
            echo "[$(ts)] skip existing $TAG"
            continue
        fi

        echo "[$(ts)] === ${TAG} ===" | tee "${LOG}"

        eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            setsid /workspace/vllm_dev_prj/bin/vllm serve ${MODEL} \
              --port ${PORT} --host 127.0.0.1 \
              --tensor-parallel-size ${TP} \
              --gpu-memory-utilization ${GPU_MEM} \
              --max-model-len ${MAX_LEN} \
              --max-num-seqs ${CONC} \
              --enable-prefix-caching \
              >> ${LOG} 2>&1 &"
        SERVE_PID=$!
        OWN_PIDS+=("$SERVE_PID")
        echo "${SERVE_PID}" > "${RUNS}/${TAG}.pid"

        READY=0
        for i in $(seq 1 240); do
            if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                READY=1; break
            fi
            sleep 5
        done
        if [[ $READY -eq 0 ]]; then
            echo "[$(ts)] $TAG: vllm not ready, skip" | tee -a "${LOG}"
            cleanup_my_pids
            continue
        fi
        echo "[$(ts)] $TAG: vllm ready" | tee -a "${LOG}"

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
            2>&1 | tee "${BENCH_PREFIX}.log"

        cleanup_my_pids
    done
done

echo "[$(ts)] Attempt 1 sweep complete."
