#!/usr/bin/env bash
# Option C Step 1 — NEO swap OOB sanity check for wd1 long-context workload.
#
# Configs:
#   vanilla — no LHC env, but uses NEO swap path (enable-neo-asymmetric)
#   lhc     — same NEO swap path + DSA hook on
#
# Verifies:
#   - engine boots + completes 64-prompt wd1 bench (no CUDA assert)
#   - count of "[NEO LHC_P4_001] swap-out OOB drop" warnings
#   - DSA hook stats present in log

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/optionC
RUNS=${BASE}/runs
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8500
TP=8
GPU_MEM=0.92
MAX_LEN=32768
INPUT_LEN=24000
OUTPUT_LEN=4096
PREFIX_LEN=200
NPROMPTS=64
CONC=8     # sanity gentle conc=8 (faster bench)
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"

CONFIGS="${CONFIGS:-vanilla lhc}"
ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

cleanup_orphans() {
    # Kill ANY process holding GPU memory (orphan workers from prior runs).
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | xargs -r kill -9 2>/dev/null || true
    sleep 4
}
cleanup_orphans

for CONFIG in $CONFIGS; do
    TAG="wd1_${CONFIG}"
    LOG="${RUNS}/${TAG}_boot.log"
    BENCH="${RUNS}/${TAG}_bench"

    case "$CONFIG" in
        vanilla)
            ENV_PRE=""
            FLAGS="--enable-neo-asymmetric"
            ;;
        lhc)
            ENV_PRE="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096 VLLM_LHC_AMX_C3=1"
            FLAGS="--enable-neo-asymmetric"
            ;;
    esac

    echo "[$(ts)] === ${TAG} ===" | tee "${LOG}"

    pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 3

    eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        nohup /workspace/vllm_dev_prj/bin/vllm serve ${MODEL} \
          --port ${PORT} --host 127.0.0.1 \
          --tensor-parallel-size ${TP} \
          --gpu-memory-utilization ${GPU_MEM} \
          --max-model-len ${MAX_LEN} \
          --max-num-seqs ${CONC} \
          --enable-prefix-caching \
          --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' \
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
        echo "[$(ts)] $TAG: vllm not ready, abort" | tee -a "${LOG}"
        pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 || true
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

echo "[$(ts)] sanity done."
