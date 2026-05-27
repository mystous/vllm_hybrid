#!/usr/bin/env bash
# SUB_191 — IDE_018 / side-channel triple-stack superposition test
# Stack 3 levers concurrently:
#   - SUB_183 NUMA pin (vanilla node0 + cores 0-49 / trident node1 + cores 56-105)
#   - SUB_188 softmax precompute (cores 80-87, 8 OMP worker, 100ms cycle)
#   - SUB_190 tokenize worker (cores 88-95, 8 OMP worker, 20ms cycle)
# Qwen 2.5 32B TP=4×2 AGSD × 500p × 32 max-tokens × 32 concurrency
# 3 mix × 3 scenario × OFF vs ON × 1-run = 9 cells × 2 = 18 measurements
# OFF = no stack (vanilla taskset only, no NUMA bind, no side-channel)
# ON  = 3-lever stack
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_018_phase_burst/SUB_191_side_channel_stack
mkdir -p "${BASE}/logs" "${BASE}/measurements/off" "${BASE}/measurements/on"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

# Reuse SUB_188 / SUB_190 binaries (do NOT rebuild)
PRECOMPUTE_BIN=/workspace/vllm_hybrid/shadow_assists/features/IDE_018_phase_burst/SUB_188_side_channel_precompute/build/side_channel_precompute
TOK_BIN=/workspace/vllm_hybrid/shadow_assists/features/IDE_016_avx512_amx_pool/SUB_190_async_tokenizer_worker/build/async_tokenizer_worker

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

# ON mode does NOT modify vLLM ENV — side-channel only.
unset VLLM_USE_PHASE_BURST VLLM_PHASE_BURST_DUMMY_FILL || true

MODEL="Qwen/Qwen2.5-32B-Instruct"

# Core split (both modes use same taskset core split for vllm)
VANILLA_CORES="0-49"
TRIDENT_CORES="56-105"
MONITOR_CORES="0-49,56-105"

# NUMA bind only applied in ON mode (SUB_183 lever)
NUMACTL_V=""
NUMACTL_T=""
if [ "${MODE}" = "on" ]; then
    NUMA_NODES=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')
    if [ "${NUMA_NODES}" = "2" ] || [ "${NUMA_NODES}" -ge 2 ] 2>/dev/null; then
        NUMACTL_V="numactl --membind=0 --cpunodebind=0"
        NUMACTL_T="numactl --membind=1 --cpunodebind=1"
        NUMA_APPLIED="yes"
    else
        NUMA_APPLIED="no_single_node"
    fi
else
    NUMA_APPLIED="off"
fi
TASKSET_V="taskset -c ${VANILLA_CORES}"
TASKSET_T="taskset -c ${TRIDENT_CORES}"
TASKSET_MON="taskset -c ${MONITOR_CORES}"

# Stack report
{
    echo "mode=${MODE}"
    echo "lever_numa_pin=${NUMA_APPLIED}"
    if [ "${MODE}" = "on" ]; then
        echo "lever_softmax=cores 80-87 / 8 OMP worker / 100ms cycle"
        echo "lever_tokenize=cores 88-95 / 8 OMP worker / 20ms cycle"
    else
        echo "lever_softmax=off"
        echo "lever_tokenize=off"
    fi
    echo "vanilla_wrap='${NUMACTL_V} ${TASKSET_V}'"
    echo "trident_wrap='${NUMACTL_T} ${TASKSET_T}'"
    echo ""
    echo "--- numactl --hardware ---"
    numactl --hardware 2>&1
} > "${BASE}/logs/stack_${SUFFIX}.txt"

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_191 mode=${MODE} starting" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "  softmax_bin=${PRECOMPUTE_BIN}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "  tokenize_bin=${TOK_BIN}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ── monitor ────────────────────────────────────────────────────────
${TASKSET_MON} "$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor_${SUFFIX}" --interval 0.5 > "${BASE}/logs/monitor_${SUFFIX}.log" 2>&1 &
MON_PID=$!
echo "[monitor] pid=${MON_PID}" >> "${BASE}/logs/main_${SUFFIX}.log"

# ── vllm vanilla (GPU 0-3 / NUMA node0 if ON) ──────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ${NUMACTL_V} ${TASKSET_V} "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    > "${BASE}/logs/vanilla_${SUFFIX}.log" 2>&1 &
V_PID=$!

# ── vllm trident (GPU 4-7 / NUMA node1 if ON) ──────────────────────
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup ${NUMACTL_T} ${TASKSET_T} "${VLLM}" serve "${MODEL}" \
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

# ── side-channel workers (ON only) ─────────────────────────────────
PC_PID=""
TK_PID=""
if [ "${MODE}" = "on" ]; then
    if [ ! -x "${PRECOMPUTE_BIN}" ]; then
        echo "[$(ts)] softmax binary missing — abort" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
        kill ${MON_PID} 2>/dev/null
        exit 1
    fi
    if [ ! -x "${TOK_BIN}" ]; then
        echo "[$(ts)] tokenize binary missing — abort" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
        kill ${MON_PID} 2>/dev/null
        exit 1
    fi
    # Softmax precompute on cores 80-87 (8 OMP worker, mask trims hardcoded 80-95 pin)
    OMP_NUM_THREADS=8 taskset -c 80-87 nohup "${PRECOMPUTE_BIN}" \
        > "${BASE}/logs/softmax_${SUFFIX}.log" 2>&1 &
    PC_PID=$!
    echo "[$(ts)] softmax pid=${PC_PID} cores=80-87" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    sleep 1
    # Tokenize worker on cores 88-95 (8 OMP worker, mask trims hardcoded 80-95 pin)
    OMP_NUM_THREADS=8 taskset -c 88-95 nohup "${TOK_BIN}" \
        > "${BASE}/logs/tokenize_${SUFFIX}.log" 2>&1 &
    TK_PID=$!
    echo "[$(ts)] tokenize pid=${TK_PID} cores=88-95" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    sleep 2
    if ! kill -0 ${PC_PID} 2>/dev/null; then
        echo "[$(ts)] softmax failed to start" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    fi
    if ! kill -0 ${TK_PID} 2>/dev/null; then
        echo "[$(ts)] tokenize failed to start" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    fi
fi

# ── router ─────────────────────────────────────────────────────────
export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1/completions"
export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1/completions"
export AGSD_MODEL="${MODEL}" AGSD_MODEL_SIZE=qwen_7b
export AGSD_CLASSIFIER_WORKERS=4 AGSD_ROUTER_PORT=8000
cd /tmp && nohup ${TASKSET_MON} "$PY" sub094_router.py > "${BASE}/logs/router_${SUFFIX}.log" 2>&1 &
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

# ── side-channel stats (ON only) ───────────────────────────────────
if [ "${MODE}" = "on" ]; then
    if [ -n "${PC_PID}" ]; then
        echo "[$(ts)] stopping softmax pid=${PC_PID}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
        kill -TERM ${PC_PID} 2>/dev/null
    fi
    if [ -n "${TK_PID}" ]; then
        echo "[$(ts)] stopping tokenize pid=${TK_PID}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
        kill -TERM ${TK_PID} 2>/dev/null
    fi
    sleep 2
    kill -9 ${PC_PID} ${TK_PID} 2>/dev/null
    echo "[$(ts)] softmax stats:" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    tail -5 "${BASE}/logs/softmax_${SUFFIX}.log" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    echo "[$(ts)] tokenize stats:" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    tail -5 "${BASE}/logs/tokenize_${SUFFIX}.log" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
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
pgrep -f side_channel_precompute 2>/dev/null | xargs -r kill -9 2>/dev/null
pgrep -f async_tokenizer_worker 2>/dev/null | xargs -r kill -9 2>/dev/null
kill ${MON_PID} 2>/dev/null

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_191 mode=${MODE} done" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
