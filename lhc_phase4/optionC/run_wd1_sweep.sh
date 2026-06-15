#!/usr/bin/env bash
# Option C Step 3.1 — W-D1 5 config × 3 sweep sweep.
#
# Configs:
#   vanilla            — no LHC env, no NEO swap (true baseline)
#   lhc_always_on      — Option A static: VLLM_LHC_DSA=1 + VLLM_LHC_AMX_C3=1
#   lhc_always_off     — explicit Option A reject (baseline equivalent)
#   lhc_adaptive       — Option C dynamic (VLLM_LHC_REGIME_ADAPTIVE=1)
#   lhc_adaptive_sfx   — Option C + suffix spec decode

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
CONC=32
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"

CONFIGS="${CONFIGS:-vanilla lhc_always_on lhc_always_off lhc_adaptive lhc_adaptive_sfx}"
SWEEPS="${SWEEPS:-3}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

cleanup_orphans() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | xargs -r kill -9 2>/dev/null || true
    sleep 4
}
cleanup_orphans

for CONFIG in $CONFIGS; do
    case "$CONFIG" in
        vanilla)
            ENV_PRE=""
            FLAGS="--enable-neo-asymmetric"
            ;;
        lhc_always_on)
            ENV_PRE="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096 VLLM_LHC_AMX_C3=1"
            FLAGS="--enable-neo-asymmetric"
            ;;
        lhc_always_off)
            ENV_PRE=""
            FLAGS="--enable-neo-asymmetric"
            ;;
        lhc_adaptive)
            ENV_PRE="VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            FLAGS="--enable-neo-asymmetric"
            ;;
        lhc_adaptive_sfx)
            ENV_PRE="VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            FLAGS="--enable-neo-asymmetric --speculative-config '{\"method\":\"suffix\",\"num_speculative_tokens\":4}'"
            ;;
        *) echo "unknown config: $CONFIG"; continue ;;
    esac

    for SWEEP in $(seq 1 $SWEEPS); do
        TAG="wd1_${CONFIG}_s${SWEEP}"
        LOG="${RUNS}/${TAG}_boot.log"
        BENCH="${RUNS}/${TAG}_bench"

        if [[ -s "${BENCH}.json" ]]; then
            echo "[$(ts)] skip existing $TAG"
            continue
        fi

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
        for i in $(seq 1 240); do
            if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                READY=1; break
            fi
            sleep 5
        done
        if [[ $READY -eq 0 ]]; then
            echo "[$(ts)] $TAG: vllm not ready, skip" | tee -a "${LOG}"
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
done

echo "[$(ts)] wd1 sweep complete."
