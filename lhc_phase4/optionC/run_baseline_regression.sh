#!/usr/bin/env bash
# Option C Step 3.2 — baseline regression: 6 workloads × {vanilla, lhc_adaptive}.
#
# Hypothesis: Option C classifier detects GPU_SATURATED in baseline regime,
# routes LHC OFF -> throughput identical to vanilla (within noise).

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/optionC
RUNS=${BASE}/runs
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8500
TP=8
GPU_MEM=0.92
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"

WORKLOADS="${WORKLOADS:-sonnet chat code balanced sonnet-heavy code-heavy}"
CONFIGS="${CONFIGS:-vanilla lhc_adaptive}"
SWEEPS="${SWEEPS:-3}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

cleanup_orphans() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | xargs -r kill -9 2>/dev/null || true
    sleep 4
}
cleanup_orphans

for WORKLOAD in $WORKLOADS; do
    case "$WORKLOAD" in
        sonnet)        INPUT_LEN=512;  OUTPUT_LEN=512; PREFIX_LEN=0; NPROMPTS=500; CONC=64; MAX_LEN=4096 ;;
        chat)          INPUT_LEN=256;  OUTPUT_LEN=512; PREFIX_LEN=0; NPROMPTS=500; CONC=64; MAX_LEN=4096 ;;
        code)          INPUT_LEN=1024; OUTPUT_LEN=512; PREFIX_LEN=0; NPROMPTS=500; CONC=64; MAX_LEN=4096 ;;
        balanced)      INPUT_LEN=768;  OUTPUT_LEN=768; PREFIX_LEN=0; NPROMPTS=500; CONC=64; MAX_LEN=4096 ;;
        sonnet-heavy)  INPUT_LEN=2048; OUTPUT_LEN=2048;PREFIX_LEN=0; NPROMPTS=200; CONC=64; MAX_LEN=8192 ;;
        code-heavy)    INPUT_LEN=4096; OUTPUT_LEN=1024;PREFIX_LEN=0; NPROMPTS=200; CONC=64; MAX_LEN=8192 ;;
        *) echo "unknown workload: $WORKLOAD"; continue ;;
    esac

    for CONFIG in $CONFIGS; do
        case "$CONFIG" in
            vanilla)
                ENV_PRE=""
                FLAGS=""
                ;;
            lhc_adaptive)
                ENV_PRE="VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
                FLAGS=""
                ;;
        esac

        for SWEEP in $(seq 1 $SWEEPS); do
            TAG="bl_${WORKLOAD}_${CONFIG}_s${SWEEP}"
            LOG="${RUNS}/${TAG}_boot.log"
            BENCH="${RUNS}/${TAG}_bench"

            if [[ -s "${BENCH}.json" ]]; then
                echo "[$(ts)] skip existing $TAG"
                continue
            fi

            echo "[$(ts)] === ${TAG} ===" | tee "${LOG}"
            cleanup_orphans

            eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                nohup /workspace/vllm_dev_prj/bin/vllm serve ${MODEL} \
                  --port ${PORT} --host 127.0.0.1 \
                  --tensor-parallel-size ${TP} \
                  --gpu-memory-utilization ${GPU_MEM} \
                  --max-model-len ${MAX_LEN} \
                  --max-num-seqs ${CONC} \
                  --enable-prefix-caching \
                  ${FLAGS} \
                  >> ${LOG} 2>&1 &"
            SERVE_PID=$!
            echo "${SERVE_PID}" > "${RUNS}/${TAG}.pid"

            READY=0
            for i in $(seq 1 180); do
                if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                    READY=1; break
                fi
                sleep 5
            done
            if [[ $READY -eq 0 ]]; then
                echo "[$(ts)] $TAG: vllm not ready, skip" | tee -a "${LOG}"
                cleanup_orphans
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
                2>&1 | tee "${BENCH}.log"

            cleanup_orphans
        done
    done
done

echo "[$(ts)] baseline regression sweep complete."
