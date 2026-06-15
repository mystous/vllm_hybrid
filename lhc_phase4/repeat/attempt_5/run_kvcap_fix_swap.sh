#!/usr/bin/env bash
# REPEAT attempt 5 — kv_cap reload fix + extreme KV pressure.
#
# Code change applied: gpu_model_runner.py LHC_P4_001 OOB guard now
# prefers cache_config.num_gpu_blocks (production value) over stale
# kv_caches[0].shape[0] from profiling.
#
# Workload: ext1 (4096 in, 1024 out, conc=256, mem=0.20) — same as eag1
# but cudagraph enabled so baseline doesn't tank.

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/repeat/attempt_5
mkdir -p "${BASE}/runs"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT=8500
TP=8
SWEEPS="${SWEEPS:-2}"
WORKLOADS="${WORKLOADS:-kvcap1 kvcap2}"
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
        # mem squeezed + high conc, max-model-len just large enough to hold input+output.
        kvcap1) INPUT_LEN=4096; OUTPUT_LEN=1024; CONC=256; NPROMPTS=512;  MAX_LEN=8192;  MEM=0.20 ;;
        kvcap2) INPUT_LEN=2048; OUTPUT_LEN=2048; CONC=512; NPROMPTS=1024; MAX_LEN=4096;  MEM=0.15 ;;
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
            for i in $(seq 1 240); do
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

echo "[$(ts)] attempt_5 sweep complete."
