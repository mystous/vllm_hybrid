#!/usr/bin/env bash
# REPEAT attempt 2 — long-context + high-conc workloads × KV pool squeeze.
#
# Goal: force GPU KV pool usage > 80% so NEO swap / DSA hook actually fires.
#
# Cells: workload ∈ {longA, longB, longC, longD} × cfg ∈ {vanilla, lhc} × 3 sweeps
#   longA — input=16384 output= 2048 conc=64  prompts=128  max_len=24576  mem=0.30
#   longB — input= 4096 output= 2048 conc=256 prompts=512  max_len= 8192  mem=0.30
#   longC — input= 8192 output= 1024 conc=128 prompts=256  max_len=16384  mem=0.30
#   longD — input= 1024 output=  256 conc=512 prompts=1024 max_len= 2048  mem=0.30
#
# Hypothesis: each cell drives KV pool >= 50% → preempt → NEO swap fires.

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_2
mkdir -p "${BASE}/runs"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT=8500
TP=8
SWEEPS="${SWEEPS:-3}"
WORKLOADS="${WORKLOADS:-longA longB longC longD}"
CONFIGS="${CONFIGS:-vanilla lhc}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

SPAWNED_PIDS=()
cleanup() {
    for pid in "${SPAWNED_PIDS[@]}"; do
        kill -9 -- -"$pid" 2>/dev/null || true
        kill -9 "$pid" 2>/dev/null || true
    done
    our_serve_pids=$(pgrep -f "vllm serve.*--port ${PORT}" 2>/dev/null || true)
    for p in $our_serve_pids; do
        kill -9 "$p" 2>/dev/null || true
    done
    sleep 5
    SPAWNED_PIDS=()
}
trap cleanup EXIT

for WORKLOAD in $WORKLOADS; do
    case "$WORKLOAD" in
        longA) INPUT_LEN=16384; OUTPUT_LEN=2048; CONC=64;  NPROMPTS=128;  MAX_LEN=24576; MEM=0.30 ;;
        longB) INPUT_LEN=4096;  OUTPUT_LEN=2048; CONC=256; NPROMPTS=512;  MAX_LEN=8192;  MEM=0.30 ;;
        longC) INPUT_LEN=8192;  OUTPUT_LEN=1024; CONC=128; NPROMPTS=256;  MAX_LEN=16384; MEM=0.30 ;;
        longD) INPUT_LEN=1024;  OUTPUT_LEN=256;  CONC=512; NPROMPTS=1024; MAX_LEN=2048;  MEM=0.30 ;;
        *) echo "unknown workload: $WORKLOAD"; continue ;;
    esac

    for CFG in $CONFIGS; do
        ENV_PRE=""; FLAGS=""
        if [[ "$CFG" == "lhc" ]]; then
            ENV_PRE="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_ADAPTIVE=1"
            FLAGS="--enable-neo-asymmetric"
        fi

        for SWEEP in $(seq 1 $SWEEPS); do
            TAG="${WORKLOAD}_${CFG}_s${SWEEP}"
            LOG="${BASE}/runs/${TAG}_boot.log"
            BENCH="${BASE}/runs/${TAG}_bench"

            if [[ -s "${BENCH}.json" ]]; then
                echo "[$(ts)] skip existing $TAG"
                continue
            fi

            echo "[$(ts)] === ${TAG} input=${INPUT_LEN} output=${OUTPUT_LEN} conc=${CONC} prompts=${NPROMPTS} mem=${MEM} ===" | tee "${LOG}"
            cleanup

            eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                setsid nohup /workspace/vllm_dev_prj/bin/vllm serve ${MODEL} \
                  --port ${PORT} --host 127.0.0.1 \
                  --tensor-parallel-size ${TP} \
                  --gpu-memory-utilization ${MEM} \
                  --max-model-len ${MAX_LEN} \
                  --max-num-seqs ${CONC} \
                  --enable-prefix-caching ${FLAGS} \
                  >> ${LOG} 2>&1 &"
            SERVE_PID=$!
            SPAWNED_PIDS+=("$SERVE_PID")
            echo "${SERVE_PID}" > "${BASE}/runs/${TAG}.pid"

            READY=0
            for i in $(seq 1 180); do
                if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                    READY=1; break
                fi
                sleep 5
            done
            if [[ $READY -eq 0 ]]; then
                echo "[$(ts)] $TAG: vllm not ready, skip" | tee -a "${LOG}"
                cleanup
                continue
            fi

            /workspace/vllm_dev_prj/bin/vllm bench serve \
                --model "${MODEL}" \
                --dataset-name sonnet \
                --dataset-path /workspace/host_vllm_hybrid/benchmarks/sonnet.txt \
                --sonnet-input-len ${INPUT_LEN} \
                --sonnet-output-len ${OUTPUT_LEN} \
                --sonnet-prefix-len 0 \
                --num-prompts ${NPROMPTS} \
                --max-concurrency ${CONC} \
                --port ${PORT} \
                --save-result --result-dir "${BASE}/runs" \
                --result-filename "${TAG}_bench.json" \
                2>&1 | tee "${BENCH}.log"

            cleanup
        done
    done
done

echo "[$(ts)] attempt_2 sweep complete."
