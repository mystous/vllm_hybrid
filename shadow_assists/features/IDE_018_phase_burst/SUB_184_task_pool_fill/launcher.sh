#!/usr/bin/env bash
# SUB_184 — IDE_018 phase-burst task-pool dummy fill canonical 500p e2e
# Qwen 2.5 32B TP=4×2 AGSD × 500p × 32 max-tokens × 32 concurrency
# 3 mix × 3 scenario × OFF vs ON × 1-run = 9 cells × 2 = 18 measurements
# OFF = default vLLM (VLLM_USE_PHASE_BURST=0)
# ON  = VLLM_USE_PHASE_BURST=1 + VLLM_PHASE_BURST_DUMMY_FILL=1 (heavy C++ dummy)
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_018_phase_burst/SUB_184_task_pool_fill
mkdir -p "${BASE}/logs" "${BASE}/measurements/off" "${BASE}/measurements/on"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

MODE="${1:-off}"
if [ "${MODE}" = "chain" ]; then
    echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] chain: OFF then ON"
    bash "$0" off
    echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] sleep 20s between cycles"
    sleep 20
    bash "$0" on
    echo "[$(TZ=Asia/Seoul date '+%H:%M:%S KST')] chain done"
    exit 0
fi

SUFFIX="${MODE}"
OUT_PREFIX="${BASE}/measurements/${MODE}"

# ── ENV common ─────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

if [ "${MODE}" = "on" ]; then
    export VLLM_USE_PHASE_BURST=1
    export VLLM_PHASE_BURST_NUM_WORKERS=8
    export VLLM_PHASE_BURST_CPU_BASE=80
    export VLLM_PHASE_BURST_DUMMY_FILL=1
    export VLLM_PHASE_BURST_DUMMY_COUNT=8
    export VLLM_PHASE_BURST_DUMMY_ITERS=8
else
    unset VLLM_USE_PHASE_BURST VLLM_PHASE_BURST_DUMMY_FILL || true
fi

MODEL="Qwen/Qwen2.5-32B-Instruct"

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_184 mode=${MODE} starting" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "  VLLM_USE_PHASE_BURST=${VLLM_USE_PHASE_BURST:-unset}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "  DUMMY_FILL=${VLLM_PHASE_BURST_DUMMY_FILL:-unset} COUNT=${VLLM_PHASE_BURST_DUMMY_COUNT:-unset} ITERS=${VLLM_PHASE_BURST_DUMMY_ITERS:-unset}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "  NUM_WORKERS=${VLLM_PHASE_BURST_NUM_WORKERS:-unset} CPU_BASE=${VLLM_PHASE_BURST_CPU_BASE:-unset}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ── monitor ────────────────────────────────────────────────────────
"$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor_${SUFFIX}" --interval 0.5 > "${BASE}/logs/monitor_${SUFFIX}.log" 2>&1 &
MON_PID=$!
echo "[monitor] pid=${MON_PID}" >> "${BASE}/logs/main_${SUFFIX}.log"

# ── vllm vanilla (GPU 0-3) ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    > "${BASE}/logs/vanilla_${SUFFIX}.log" 2>&1 &
V_PID=$!

# ── vllm trident (GPU 4-7) ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup "${VLLM}" serve "${MODEL}" \
    --port 8002 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
    > "${BASE}/logs/trident_${SUFFIX}.log" 2>&1 &
T_PID=$!

BOOT_T0=$(date +%s)
echo "[$(ts)] waiting vllm (V_PID=${V_PID} T_PID=${T_PID})" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
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
    echo "[$(ts)] vllm not ready in 20 min, abort" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    kill ${MON_PID} ${V_PID} ${T_PID} 2>/dev/null
    pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
    exit 1
fi
echo "[$(ts)] vllm ready ${i}x10s (boot=${BOOT_DUR}s)" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "${BOOT_DUR}" > "${BASE}/logs/boot_${SUFFIX}_seconds.txt"

# ── router ─────────────────────────────────────────────────────────
export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1/completions"
export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1/completions"
export AGSD_MODEL="${MODEL}" AGSD_MODEL_SIZE=qwen_7b
export AGSD_CLASSIFIER_WORKERS=4 AGSD_ROUTER_PORT=8000
cd /tmp && nohup "$PY" sub094_router.py > "${BASE}/logs/router_${SUFFIX}.log" 2>&1 &
R_PID=$!
sleep 10

H=$(curl -sf -m 3 "http://127.0.0.1:8000/health" 2>/dev/null || echo "")
echo "[$(ts)] router health: ${H}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ── benchmark 3 mix ────────────────────────────────────────────────
export BENCH_MODEL="${MODEL}"
echo "[$(ts)] === ${MODE} 500p x 3 mix benchmark (max-tokens=32) ===" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
for MIX in balanced sonnet-heavy code-heavy; do
    OUT="${OUT_PREFIX}/${MIX}"
    mkdir -p "${OUT}"
    curl -sX POST http://127.0.0.1:8000/reset > /dev/null
    echo "[$(ts)] mix=${MIX}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    "$PY" /tmp/sub094_benchmark.py \
        --num-prompts 500 --max-tokens 32 --concurrency 32 \
        --mix ${MIX} --out-dir "${OUT}/" \
        > "${OUT}/bench.log" 2>&1
    curl -sf -m 3 "http://127.0.0.1:8000/stats" > "${OUT}/router_stats.json" 2>/dev/null || true
done

# ── phase-burst stats (ON only) ────────────────────────────────────
if [ "${MODE}" = "on" ]; then
    echo "[$(ts)] capture phase-burst stats" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    # we can't snapshot in-proc; rely on stderr log scrape later.
    grep -E "phase-burst|dummy" "${BASE}/logs/vanilla_${SUFFIX}.log" "${BASE}/logs/trident_${SUFFIX}.log" \
        > "${BASE}/logs/phase_burst_log_${SUFFIX}.txt" 2>/dev/null || true
fi

# ── cleanup ────────────────────────────────────────────────────────
echo "[$(ts)] cleanup" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
kill ${R_PID} 2>/dev/null
sleep 2
pgrep -f sub094_router 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 2
pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3
kill ${MON_PID} 2>/dev/null

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_184 mode=${MODE} done" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
