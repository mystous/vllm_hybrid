#!/usr/bin/env bash
# LHC Phase 3 — Task D pilot: KV-heavy workload + NEO swap firing verify.
#
# usage: ./run_kv_heavy_pilot.sh <workload> <config>
#   workload ∈ {wd1, wd3}
#   config   ∈ {vanilla, lhc_dsa}
#
# Boots a single vllm serve on GPU 0-7 (TP=8), runs `vllm bench serve` with
# the target workload, then greps the boot log for NEO swap counts + DSA
# hook stats.

set -uo pipefail
WORKLOAD="${1:-wd1}"
CONFIG="${2:-vanilla}"

BASE=/workspace/host_vllm_hybrid/lhc_phase3
mkdir -p "${BASE}/runs_D/${WORKLOAD}"
LOG="${BASE}/runs_D/${WORKLOAD}/${CONFIG}_boot.log"
BENCH_LOG="${BASE}/runs_D/${WORKLOAD}/${CONFIG}_bench.log"

export HF_HUB_OFFLINE=1
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8500

# Workload-specific defaults.
case "${WORKLOAD}" in
    wd1)
        # W-D1: long-context 32K, KV-pressure pilot
        # We push concurrency high + clamp gpu-memory-utilization to force NEO
        # swap firing. B200 184GB × 8 = 1.47TB HBM — vanilla never swaps unless
        # KV pool is artificially shrunk.
        SONNET_INPUT_LEN=24000  # ~28k input pushes KV near 32k boundary
        SONNET_OUTPUT_LEN=4096
        SONNET_PREFIX_LEN=200
        NUM_PROMPTS=64
        CONCURRENCY=32
        MAX_MODEL_LEN=32768
        MAX_NUM_SEQS=32
        GPU_MEM_UTIL=0.18   # tightens KV pool to ~16GB / GPU → guaranteed swap
        ;;
    wd3)
        # W-D3: prefix-cache heavy, conc=64
        SONNET_INPUT_LEN=12000
        SONNET_OUTPUT_LEN=512
        SONNET_PREFIX_LEN=8000  # shared 8K prefix
        NUM_PROMPTS=128
        CONCURRENCY=32
        MAX_MODEL_LEN=16384
        MAX_NUM_SEQS=64
        GPU_MEM_UTIL=0.60
        ;;
    *) echo "unknown workload: ${WORKLOAD}"; exit 1 ;;
esac

# Config-specific envs.
NEO_FLAG=""
DSA_ENV=""
case "${CONFIG}" in
    vanilla)
        ;;
    lhc_dsa)
        NEO_FLAG="--enable-neo-asymmetric"
        DSA_ENV="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096"
        ;;
    *) echo "unknown config: ${CONFIG}"; exit 1 ;;
esac

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

# Pre-clean any stale vllm processes (safe — fresh container).
pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 2

echo "[$(ts)] launching vllm: ${CONFIG} workload=${WORKLOAD}" | tee "${LOG}"
echo "[cfg] sonnet_input=${SONNET_INPUT_LEN} output=${SONNET_OUTPUT_LEN} prefix=${SONNET_PREFIX_LEN}" \
    "num_prompts=${NUM_PROMPTS} conc=${CONCURRENCY} max_model_len=${MAX_MODEL_LEN} mem_util=${GPU_MEM_UTIL}" \
    | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    VLLM_LHC_DSA="${VLLM_LHC_DSA:-0}" \
    VLLM_LHC_DSA_WQ_PER_RANK="${VLLM_LHC_DSA_WQ_PER_RANK:-0}" \
    VLLM_LHC_DSA_MIN="${VLLM_LHC_DSA_MIN:-65536}" \
    eval ${DSA_ENV} \
    nohup /workspace/vllm_dev_prj/bin/vllm serve "${MODEL}" \
        --port ${PORT} --host 127.0.0.1 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --enable-prefix-caching \
        ${NEO_FLAG} \
    >> "${LOG}" 2>&1 &
SERVE_PID=$!
echo "${SERVE_PID}" > "${BASE}/runs_D/${WORKLOAD}/${CONFIG}.pid"
PGID=$(ps -o pgid= -p ${SERVE_PID} 2>/dev/null | tr -d ' ' || echo "")
echo "[$(ts)] serve PID=${SERVE_PID} PGID=${PGID}" | tee -a "${LOG}"

# Wait for ready
echo "[$(ts)] waiting for vllm /health..." | tee -a "${LOG}"
READY=0
for i in $(seq 1 90); do
    sleep 10
    if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        READY=1; break
    fi
done
if [ "${READY}" = "0" ]; then
    echo "[$(ts)] vllm not ready in 15min — abort" | tee -a "${LOG}"
    [ -n "${PGID}" ] && kill -9 -${PGID} 2>/dev/null
    pgrep -f "VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null
    exit 1
fi
echo "[$(ts)] vllm ready (${i}x10s = ${i}0s)" | tee -a "${LOG}"

# Bench
echo "[$(ts)] running bench (workload=${WORKLOAD} cfg=${CONFIG})" | tee -a "${LOG}"
/workspace/vllm_dev_prj/bin/python -m vllm.entrypoints.cli.main bench serve \
    --backend vllm --model "${MODEL}" \
    --host 127.0.0.1 --port ${PORT} \
    --dataset-name sonnet --dataset-path /workspace/host_vllm_hybrid/benchmarks/sonnet.txt \
    --sonnet-input-len ${SONNET_INPUT_LEN} \
    --sonnet-output-len ${SONNET_OUTPUT_LEN} \
    --sonnet-prefix-len ${SONNET_PREFIX_LEN} \
    --num-prompts ${NUM_PROMPTS} \
    --max-concurrency ${CONCURRENCY} \
    --save-result --result-dir "${BASE}/runs_D/${WORKLOAD}/" \
    --result-filename "${CONFIG}_bench.json" \
    > "${BENCH_LOG}" 2>&1

echo "[$(ts)] bench done; collecting NEO+DSA stats" | tee -a "${LOG}"

# Hook stats from the boot log
NEO_SWAP_OUT=$(grep -c "\[NEO\] swap-out:" "${LOG}" 2>/dev/null | head -1 | tr -d '\n' || echo 0)
NEO_SWAP_IN=$(grep -c "\[NEO\] swap-in:" "${LOG}" 2>/dev/null | head -1 | tr -d '\n' || echo 0)
NEO_DRAIN_FAIL=$(grep -c "\[NEO\] swap_out drain.*scatter failed" "${LOG}" 2>/dev/null | head -1 | tr -d '\n' || echo 0)
DSA_ENABLED=$(grep -c "\[LHC DSA\] lane ENABLED" "${LOG}" 2>/dev/null | head -1 | tr -d '\n' || echo 0)
KV_USAGE_MAX=$(grep -oP "GPU KV cache usage: \K[0-9.]+" "${LOG}" 2>/dev/null | sort -n | tail -1 || echo 0)

cat > "${BASE}/runs_D/${WORKLOAD}/${CONFIG}_hook_stats.json" <<EOF
{
  "config": "${CONFIG}",
  "workload": "${WORKLOAD}",
  "neo_swap_out_log_count": ${NEO_SWAP_OUT},
  "neo_swap_in_log_count": ${NEO_SWAP_IN},
  "neo_drain_scatter_fail": ${NEO_DRAIN_FAIL},
  "dsa_lane_enabled_workers": ${DSA_ENABLED},
  "kv_usage_pct_max": ${KV_USAGE_MAX}
}
EOF
cat "${BASE}/runs_D/${WORKLOAD}/${CONFIG}_hook_stats.json" | tee -a "${LOG}"

# Cleanup
echo "[$(ts)] killing serve PGID=${PGID}" | tee -a "${LOG}"
[ -n "${PGID}" ] && kill -9 -${PGID} 2>/dev/null
sleep 3
pgrep -f "VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null
pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5

# Orphan GPU procs
for g in 0 1 2 3 4 5 6 7; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $g 2>/dev/null | \
        awk -F, '{print $1}' | xargs -r kill -9 2>/dev/null
done
sleep 2

echo "[$(ts)] cycle done" | tee -a "${LOG}"
