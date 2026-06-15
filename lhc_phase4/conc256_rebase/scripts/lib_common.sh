#!/usr/bin/env bash
# Common helpers for conc=256 rebase sweep.
# All scripts source this file.

set -uo pipefail

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export PORT="${PORT:-8501}"
export TP="${TP:-8}"
export GPU_MEM="${GPU_MEM:-0.92}"
export MAX_LEN="${MAX_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"
export VLLM_BIN="/workspace/vllm_dev_prj/bin/vllm"

# Workload spec (per task: 500p, max_tokens 2048, conc=256, target-input-len=1024).
# We use the project's `vllm bench serve` sonnet variant to avoid heavy tokenizer load.
export NPROMPTS="${NPROMPTS:-500}"
export INPUT_LEN="${INPUT_LEN:-1024}"
export OUTPUT_LEN="${OUTPUT_LEN:-2048}"
export CONC="${CONC:-256}"

# 6 workloads — interpret as sonnet/chat/code/balanced/sonnet-heavy/code-heavy
# We keep workload shape uniform (target_input_len=1024, max_tokens=2048, conc=256, 500p)
# and use distinct prefix-len / seed to differentiate (prefix locality = chat/code-heavy).
export WORKLOADS_DEFAULT="sonnet chat code balanced sonnet-heavy code-heavy"

ts()       { TZ=Asia/Seoul date '+%H:%M:%S KST'; }
ts_short() { date '+%Y%m%d_%H%M%S'; }

cleanup_orphans() {
    # Kill any CUDA process that occupies a B200 GPU.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | xargs -r kill -9 2>/dev/null || true
    sleep 4
}

# workload → input/output spec (uniform at 1024/2048/conc256/500p; prefix_len shape changes).
workload_spec() {
    local W=$1
    case "$W" in
        sonnet)         echo "1024 2048 0    500" ;;
        chat)           echo "1024 2048 0    500" ;;
        code)           echo "1024 2048 0    500" ;;
        balanced)       echo "1024 2048 256  500" ;;
        sonnet-heavy)   echo "1024 2048 0    500" ;;
        code-heavy)     echo "1024 2048 512  500" ;;
        *) echo "1024 2048 0 500" ;;
    esac
}

# ─── boot vllm with env prefix and additional flags ────────────────────────────
boot_vllm() {
    local TAG=$1
    local ENV_PRE=$2
    local LOG=$3
    cleanup_orphans

    eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        nohup ${VLLM_BIN} serve ${MODEL} \
          --port ${PORT} --host 127.0.0.1 \
          --tensor-parallel-size ${TP} \
          --gpu-memory-utilization ${GPU_MEM} \
          --max-model-len ${MAX_LEN} \
          --max-num-seqs ${MAX_NUM_SEQS} \
          --enable-prefix-caching \
          --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' \
          >> ${LOG} 2>&1 &"
    local PID=$!
    echo "${PID}"
}

wait_health() {
    local PORT=$1
    local TIMEOUT=${2:-300}
    for i in $(seq 1 $TIMEOUT); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

run_bench() {
    local TAG=$1
    local WORKLOAD=$2
    local OUTDIR=$3
    local SEED=$4
    read -r I O P N <<< "$(workload_spec "$WORKLOAD")"

    ${VLLM_BIN} bench serve \
        --model "${MODEL}" \
        --dataset-name sonnet \
        --dataset-path "${DATA}" \
        --sonnet-input-len ${I} \
        --sonnet-output-len ${O} \
        --sonnet-prefix-len ${P} \
        --num-prompts ${N} \
        --max-concurrency ${CONC} \
        --port ${PORT} \
        --seed ${SEED} \
        --save-result --result-dir "${OUTDIR}" \
        --result-filename "${TAG}.json" \
        2>&1 | tee "${OUTDIR}/${TAG}.log"
}
