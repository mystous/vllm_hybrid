#!/usr/bin/env bash
# LHC Phase 4 — REPEAT attempt 1: force KV pool pressure via gpu_memory_utilization sweep.
#
# Hypothesis: NEO swap path / DSA hook misses fire because B200 1464GB HBM dwarfs
# baseline KV demand. Compressing the GPU memory budget should activate the
# swap path on the same workload.
#
# Cells: mem_util ∈ {0.30, 0.25, 0.20, 0.15} × config ∈ {vanilla, lhc} × 3 sweeps.
# Workload: sonnet 512/512 conc=64 prompts=500 (baseline-equivalent).
#
# Outputs: lhc_phase4/repeat/attempt_1/runs/<mem>_<cfg>_<sweep>_bench.{json,log}
#          lhc_phase4/repeat/attempt_1/runs/<mem>_<cfg>_<sweep>_boot.log

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_1
mkdir -p "${BASE}/runs"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT=8500
TP=8
MAX_LEN=4096
INPUT_LEN=512
OUTPUT_LEN=512
NPROMPTS=500
CONC=64
SWEEPS="${SWEEPS:-3}"

MEMS="${MEMS:-0.30 0.25 0.20 0.15}"
CONFIGS="${CONFIGS:-vanilla lhc}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

# Track self-spawned vllm PIDs to avoid killing unrelated processes.
SPAWNED_PIDS=()
cleanup() {
    for pid in "${SPAWNED_PIDS[@]}"; do
        # kill the process group of the nohup'd vllm serve
        kill -9 -- -"$pid" 2>/dev/null || true
        kill -9 "$pid" 2>/dev/null || true
    done
    # Also kill any child worker python procs that survived
    if [[ ${#SPAWNED_PIDS[@]} -gt 0 ]]; then
        # Find compute apps whose parent chain points to our nohup pids,
        # by matching VLLM::Worker process names of our started instance.
        # Use --filter on command line to be conservative.
        local our_serve_pids
        our_serve_pids=$(pgrep -f "vllm serve.*--port ${PORT}" 2>/dev/null || true)
        for p in $our_serve_pids; do
            kill -9 "$p" 2>/dev/null || true
        done
    fi
    sleep 5
    SPAWNED_PIDS=()
}
trap cleanup EXIT
# Initial state check — assume environment is clean.

for MEM in $MEMS; do
    for CFG in $CONFIGS; do
        ENV_PRE=""
        FLAGS=""
        if [[ "$CFG" == "lhc" ]]; then
            ENV_PRE="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_ADAPTIVE=1"
            FLAGS="--enable-neo-asymmetric"
        fi

        for SWEEP in $(seq 1 $SWEEPS); do
            TAG="mem${MEM}_${CFG}_s${SWEEP}"
            LOG="${BASE}/runs/${TAG}_boot.log"
            BENCH="${BASE}/runs/${TAG}_bench"

            if [[ -s "${BENCH}.json" ]]; then
                echo "[$(ts)] skip existing $TAG"
                continue
            fi

            echo "[$(ts)] === ${TAG} ===" | tee "${LOG}"
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

            # Wait for ready (max 600s).
            READY=0
            for i in $(seq 1 120); do
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

echo "[$(ts)] attempt_1 sweep complete."
