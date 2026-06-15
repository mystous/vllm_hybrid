#!/usr/bin/env bash
# LHC Phase 4 Option C — Path 1: AMX C3 prefix hash chain.
#
# Workload selection: prefix-heavy chat (high prefix-cache hit rate so the
# hash chain dominates scheduler CPU time) + sonnet baseline regression.
#
# Configs:
#   - vanilla            : SHA-256 block-hash (vLLM default).
#   - lhc_amx_c3_prefix  : FNV-1a chain via _lhc_amx_c3_block_hash().
#
# Gate: throughput +5% AND hook calls > 100 / min AND cache hit Δ ≤ 1pp.

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/option_C_bypass/path_1
RUNS=${BASE}/runs
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8512
TP=8
GPU_MEM=0.92
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"

WORKLOADS="${WORKLOADS:-chat_prefix sonnet}"
CONFIGS="${CONFIGS:-vanilla lhc_amx_c3_prefix}"
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
        # prefix-heavy chat: 2048-token shared prefix + 256 user + 512 output
        # so prefix-cache hit rate ≈ 80% across 500 prompts.
        chat_prefix)  INPUT_LEN=2304; OUTPUT_LEN=512; PREFIX_LEN=2048; NPROMPTS=500; CONC=64; MAX_LEN=4096 ;;
        sonnet)       INPUT_LEN=512;  OUTPUT_LEN=512; PREFIX_LEN=0;    NPROMPTS=500; CONC=64; MAX_LEN=4096 ;;
        *) echo "unknown workload: $WORKLOAD"; continue ;;
    esac

    for CONFIG in $CONFIGS; do
        case "$CONFIG" in
            vanilla)
                ENV_PRE=""
                ;;
            lhc_amx_c3_prefix)
                ENV_PRE="VLLM_LHC_AMX_C3_PREFIX=1"
                ;;
        esac

        for SWEEP in $(seq 1 $SWEEPS); do
            TAG="p1_${WORKLOAD}_${CONFIG}_s${SWEEP}"
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

echo "[$(ts)] Path 1 sweep complete."
