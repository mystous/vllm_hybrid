#!/usr/bin/env bash
# SUB_181 — IDE_019 / TSK_037 Jacobi 4-method router integration + canonical 500p e2e
# Qwen 2.5 32B TP=4×2 / AGSD 4-method × 500p × 32 max-tokens × 32 concurrency
# 3 mix × 1-run
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter/SUB_181_jacobi_router_e2e
mkdir -p "${BASE}/logs" "${BASE}/measurements/4method_500p" "${BASE}/measurements/3method_500p"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""

# pthread EAGAIN 회피
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-32B-Instruct"
SUFFIX="${1:-4method}"
USE_JACOBI="${2:-1}"
OUT_PREFIX="${BASE}/measurements/${SUFFIX}_500p"
mkdir -p "${OUT_PREFIX}"

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_181 starting suffix=${SUFFIX} USE_JACOBI=${USE_JACOBI}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

"$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor_${SUFFIX}" --interval 0.5 > "${BASE}/logs/monitor_${SUFFIX}.log" 2>&1 &
MON_PID=$!
echo "[monitor] pid=${MON_PID}" >> "${BASE}/logs/main_${SUFFIX}.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    > "${BASE}/logs/vanilla_${SUFFIX}.log" 2>&1 &
V_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup "${VLLM}" serve "${MODEL}" \
    --port 8002 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
    > "${BASE}/logs/trident_${SUFFIX}.log" 2>&1 &
T_PID=$!

BOOT_T0=$(date +%s)
echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] waiting vllm (V_PID=${V_PID} T_PID=${T_PID})" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
READY=0
for i in $(seq 1 120); do
    sleep 10
    V_OK=$(curl -sf -m 3 "http://127.0.0.1:8001/v1/models" 2>/dev/null | head -c 30 || echo "")
    T_OK=$(curl -sf -m 3 "http://127.0.0.1:8002/v1/models" 2>/dev/null | head -c 30 || echo "")
    if [ -n "$V_OK" ] && [ -n "$T_OK" ]; then
        READY=1
        break
    fi
done
BOOT_T1=$(date +%s)
BOOT_DUR=$(( BOOT_T1 - BOOT_T0 ))
if [ "${READY}" = "0" ]; then
    echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] vllm not ready in 20 min, abort" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    kill ${MON_PID} ${V_PID} ${T_PID} 2>/dev/null
    pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
    exit 1
fi
echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] vllm ready ${i}x10s (boot=${BOOT_DUR}s)" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "${BOOT_DUR}" > "${BASE}/logs/boot_${SUFFIX}_seconds.txt"

# launch SUB_181 router (4-method with cpu_jacobi gated by AGSD_USE_JACOBI)
export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1/completions"
export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1/completions"
export AGSD_MODEL="${MODEL}" AGSD_MODEL_SIZE=qwen_7b
export AGSD_CLASSIFIER_WORKERS=4 AGSD_ROUTER_PORT=8000
export AGSD_USE_JACOBI="${USE_JACOBI}"
export AGSD_JACOBI_K=5
export AGSD_JACOBI_THREADS=64
export AGSD_JACOBI_HIDDEN=5120
export AGSD_JACOBI_VOCAB=152064
export AGSD_JACOBI_CONCURRENCY=2
cd "${BASE}/src" && nohup taskset -c 0-99 "$PY" sub181_router.py > "${BASE}/logs/router_${SUFFIX}.log" 2>&1 &
R_PID=$!
sleep 10

# probe router health
H=$(curl -sf -m 3 "http://127.0.0.1:8000/health" 2>/dev/null || echo "")
echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] router health: ${H}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

export BENCH_MODEL="${MODEL}"
echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] === canonical ${SUFFIX} 500p × 3 mix benchmark (max-tokens=32) ===" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
for MIX in balanced sonnet-heavy code-heavy; do
    OUT="${OUT_PREFIX}/${MIX}"
    mkdir -p "${OUT}"
    curl -sX POST http://127.0.0.1:8000/reset > /dev/null
    echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] mix=${MIX}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    "$PY" /tmp/sub094_benchmark.py \
        --num-prompts 500 --max-tokens 32 --concurrency 32 \
        --mix ${MIX} --out-dir "${OUT}/" \
        > "${OUT}/bench.log" 2>&1
    # snapshot router stats for jacobi accounting
    curl -sf -m 3 "http://127.0.0.1:8000/stats" > "${OUT}/router_stats.json" 2>/dev/null
done

# cleanup
echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] cleanup" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
kill ${R_PID} 2>/dev/null
sleep 2
pgrep -f sub181_router 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 2
pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3
kill ${MON_PID} 2>/dev/null

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_181 ${SUFFIX} done" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
