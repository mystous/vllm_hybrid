#!/usr/bin/env bash
# SUB_183 — IDE_017 NUMA-aware vllm instance pinning canonical 500p e2e
# Qwen 2.5 32B TP=4×2 AGSD × 500p × 32 max-tokens × 32 concurrency
# 3 mix × 3 scenario × {OFF taskset-only, ON numactl+taskset} = 9 cells × 2 = 18 measurements
# 1-run default
#
# NUMA topology (host):
#   node0 cpus 0-55 (phys) + 112-167 (HT) — memory 1031 GB
#   node1 cpus 56-111 (phys) + 168-223 (HT) — memory 1032 GB
#   node distance 0<->1 = 21 (10 local), GPU 0-3 ↔ node0, GPU 4-7 ↔ node1 (Phase A SUB_113)
#
# OFF: taskset core split only (no numactl) — memory allocation default OS policy
#   vanilla: taskset -c 0-49 (50 cores on node0 physical)
#   trident: taskset -c 56-105 (50 cores on node1 physical)
# ON: numactl --membind + --cpunodebind + taskset core split
#   vanilla: numactl --membind=0 --cpunodebind=0 + taskset -c 0-49
#   trident: numactl --membind=1 --cpunodebind=1 + taskset -c 56-105
# Total 100 phys cores in both modes (HT siblings 112-223 unused).
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/SUB_183_numa_pinning_canonical
mkdir -p "${BASE}/logs" "${BASE}/measurements/off" "${BASE}/measurements/on"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""

# pthread EAGAIN 회피 (both modes — boot-stability, not the NUMA lever)
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-32B-Instruct"
MODE="${1:-off}"   # 'off' or 'on'
SUFFIX="${MODE}"
OUT_PREFIX="${BASE}/measurements/${MODE}"

# core sets (both modes identical, only numactl differs)
VANILLA_CORES="0-49"     # 50 phys cores on NUMA node0
TRIDENT_CORES="56-105"   # 50 phys cores on NUMA node1
MONITOR_CORES="0-49,56-105"  # union, observability host

# NUMA pinning report — what actually applied
NUMA_REPORT="${BASE}/logs/numa_${SUFFIX}.txt"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_183 mode=${MODE} starting" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ---------- Determine wrappers ----------
NUMACTL_V=""
NUMACTL_T=""
NUMA_NODES=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')

if [ "${MODE}" = "on" ]; then
    if [ "${NUMA_NODES}" = "2" ] || [ "${NUMA_NODES}" -ge 2 ] 2>/dev/null; then
        NUMACTL_V="numactl --membind=0 --cpunodebind=0"
        NUMACTL_T="numactl --membind=1 --cpunodebind=1"
        NUMA_APPLIED="yes"
    else
        # Single-node — no-op
        NUMA_APPLIED="no_single_node"
    fi
else
    NUMA_APPLIED="off"
fi

TASKSET_V="taskset -c ${VANILLA_CORES}"
TASKSET_T="taskset -c ${TRIDENT_CORES}"
TASKSET_MON="taskset -c ${MONITOR_CORES}"

# Write NUMA report
{
    echo "mode=${MODE}"
    echo "numa_nodes=${NUMA_NODES}"
    echo "numa_applied=${NUMA_APPLIED}"
    echo "vanilla_cores=${VANILLA_CORES}"
    echo "trident_cores=${TRIDENT_CORES}"
    echo "vanilla_wrap='${NUMACTL_V} ${TASKSET_V}'"
    echo "trident_wrap='${NUMACTL_T} ${TASKSET_T}'"
    echo ""
    echo "--- numactl --hardware ---"
    numactl --hardware 2>&1
} > "${NUMA_REPORT}"

# ---------- monitor (no NUMA pin, taskset only across both nodes) ----------
${TASKSET_MON} "$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor_${SUFFIX}" --interval 0.5 > "${BASE}/logs/monitor_${SUFFIX}.log" 2>&1 &
MON_PID=$!
echo "[monitor] pid=${MON_PID}" >> "${BASE}/logs/main_${SUFFIX}.log"

# ---------- vllm vanilla (NUMA node0, GPU 0-3) ----------
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ${NUMACTL_V} ${TASKSET_V} "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    > "${BASE}/logs/vanilla_${SUFFIX}.log" 2>&1 &
V_PID=$!

# ---------- vllm trident (NUMA node1, GPU 4-7) ----------
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

# ---------- router (3-method, sub094) — no NUMA pin, taskset union ----------
export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1/completions"
export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1/completions"
export AGSD_MODEL="${MODEL}" AGSD_MODEL_SIZE=qwen_7b
export AGSD_CLASSIFIER_WORKERS=4 AGSD_ROUTER_PORT=8000
cd /tmp && nohup ${TASKSET_MON} "$PY" sub094_router.py > "${BASE}/logs/router_${SUFFIX}.log" 2>&1 &
R_PID=$!
sleep 10

H=$(curl -sf -m 3 "http://127.0.0.1:8000/health" 2>/dev/null || echo "")
echo "[$(ts)] router health: ${H}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ---------- benchmark ----------
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

# ---------- cleanup ----------
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

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_183 mode=${MODE} done" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
