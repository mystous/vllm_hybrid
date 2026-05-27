#!/usr/bin/env bash
# SUB_186 — IDE_020 environment stack (SUB_182 isolation + SUB_183 NUMA pin) canonical 500p e2e
# Qwen 2.5 32B TP=4×2 AGSD × 500p × 32 max-tokens × 32 concurrency
# 3 mix × 3 scenario × {OFF default, ON full stack} = 9 cells × 2 = 18 measurements
# 1-run default
#
# OFF (default): no isolation, no NUMA pin
#   vanilla = unrestricted (no taskset, no numactl)
#   trident = unrestricted (no taskset, no numactl)
#
# ON (SUB_182 + SUB_183 stack 동시):
#   cgroup v2 `/sys/fs/cgroup/sub186_on/cpuset.cpus = 0-99` (100 phys cores, no HT)
#   hugepages 4 × 2 MB (echo 4 > /proc/sys/vm/nr_hugepages)
#   VLLM_USE_HUGEPAGES=1
#   vanilla: numactl --membind=0 --cpunodebind=0 taskset -c 0-49 (50 phys cores on node0)
#   trident: numactl --membind=1 --cpunodebind=1 taskset -c 56-105 (50 phys cores on node1)
#   monitor/router: taskset -c 0-49,56-105 union
#
# NUMA topology (SUB_183 reference):
#   node0 cpus 0-55 (phys) + 112-167 (HT) — memory 1031 GB
#   node1 cpus 56-111 (phys) + 168-223 (HT) — memory 1032 GB
#   GPU 0-3 ↔ node0, GPU 4-7 ↔ node1 (Phase A SUB_113)
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_020_production_isolation/SUB_186_env_stack
mkdir -p "${BASE}/logs" "${BASE}/measurements/off" "${BASE}/measurements/on"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""

# pthread EAGAIN 회피 (both modes — boot-stability)
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-32B-Instruct"
MODE="${1:-off}"   # 'off' or 'on'
SUFFIX="${MODE}"
OUT_PREFIX="${BASE}/measurements/${MODE}"

# Core sets (ON only — OFF leaves unrestricted)
VANILLA_CORES="0-49"     # 50 phys cores on NUMA node0
TRIDENT_CORES="56-105"   # 50 phys cores on NUMA node1
MONITOR_CORES="0-49,56-105"  # union, observability host

# Isolation report — what actually applied
ISOLATION_REPORT="${BASE}/logs/isolation_${SUFFIX}.txt"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_186 mode=${MODE} starting" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ---------- Apply isolation + NUMA stack (ON only) ----------
WRAP_V=""        # vanilla wrap
WRAP_T=""        # trident wrap
WRAP_MON=""      # monitor wrap
WRAP_RTR=""      # router wrap
CGROUP_OK="no"
HUGEPAGES_OK="no"
HUGEPAGES_ALLOC=0
NUMA_APPLIED="off"
TASKSET_APPLIED="no"
CG_PATH=/sys/fs/cgroup/sub186_on
NUMA_NODES=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')

if [ "${MODE}" = "on" ]; then
    # 1) hugepages (SUB_182 lever)
    if [ -w /proc/sys/vm/nr_hugepages ]; then
        echo 4 > /proc/sys/vm/nr_hugepages
        HUGEPAGES_ALLOC=$(cat /proc/sys/vm/nr_hugepages)
        if [ "${HUGEPAGES_ALLOC}" -ge 1 ]; then
            HUGEPAGES_OK="yes"
        fi
        export VLLM_USE_HUGEPAGES=1
    fi

    # 2) cgroup v2 cpuset 0-99 (SUB_182 lever)
    if [ -d /sys/fs/cgroup ] && [ -w /sys/fs/cgroup ]; then
        echo "+cpuset" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
        mkdir -p "${CG_PATH}" 2>/dev/null
        if [ -w "${CG_PATH}/cpuset.cpus" ]; then
            echo "0-99" > "${CG_PATH}/cpuset.cpus" 2>/dev/null
            EFF=$(cat "${CG_PATH}/cpuset.cpus.effective" 2>/dev/null)
            if [ "${EFF}" = "0-99" ]; then
                CGROUP_OK="yes"
            fi
        fi
    fi

    # 3) NUMA pin (SUB_183 lever)
    if [ "${NUMA_NODES}" = "2" ] || [ "${NUMA_NODES}" -ge 2 ] 2>/dev/null; then
        NUMACTL_V="numactl --membind=0 --cpunodebind=0"
        NUMACTL_T="numactl --membind=1 --cpunodebind=1"
        NUMA_APPLIED="yes"
    else
        NUMACTL_V=""
        NUMACTL_T=""
        NUMA_APPLIED="no_single_node"
    fi

    # 4) taskset core split (SUB_182 + SUB_183 공통 lever, 100 phys core max)
    TASKSET_V="taskset -c ${VANILLA_CORES}"
    TASKSET_T="taskset -c ${TRIDENT_CORES}"
    TASKSET_MON="taskset -c ${MONITOR_CORES}"
    TASKSET_APPLIED="yes"

    # Compose wrappers (NUMA + taskset both applied per role)
    WRAP_V="${NUMACTL_V} ${TASKSET_V}"
    WRAP_T="${NUMACTL_T} ${TASKSET_T}"
    WRAP_MON="${TASKSET_MON}"
    WRAP_RTR="${TASKSET_MON}"
fi

# Write isolation report
{
    echo "mode=${MODE}"
    echo "cgroup_ok=${CGROUP_OK}"
    echo "cgroup_path=${CG_PATH}"
    echo "hugepages_ok=${HUGEPAGES_OK}"
    echo "hugepages_alloc=${HUGEPAGES_ALLOC}"
    echo "VLLM_USE_HUGEPAGES=${VLLM_USE_HUGEPAGES:-unset}"
    echo "numa_nodes=${NUMA_NODES}"
    echo "numa_applied=${NUMA_APPLIED}"
    echo "taskset_applied=${TASKSET_APPLIED}"
    echo "vanilla_wrap='${WRAP_V}'"
    echo "trident_wrap='${WRAP_T}'"
    echo "monitor_wrap='${WRAP_MON}'"
    echo "router_wrap='${WRAP_RTR}'"
    if [ "${MODE}" = "on" ] && [ "${CGROUP_OK}" = "yes" ]; then
        echo "cgroup_cpuset.cpus.effective=$(cat ${CG_PATH}/cpuset.cpus.effective 2>/dev/null)"
    fi
    echo ""
    echo "--- numactl --hardware ---"
    numactl --hardware 2>&1
} > "${ISOLATION_REPORT}"

# Helper: optionally place a PID into cgroup
place_in_cgroup() {
    local pid=$1
    if [ "${CGROUP_OK}" = "yes" ] && [ -n "${pid}" ]; then
        echo "${pid}" > "${CG_PATH}/cgroup.procs" 2>/dev/null || true
    fi
}

# ---------- monitor ----------
${WRAP_MON} "$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor_${SUFFIX}" --interval 0.5 > "${BASE}/logs/monitor_${SUFFIX}.log" 2>&1 &
MON_PID=$!
echo "[monitor] pid=${MON_PID}" >> "${BASE}/logs/main_${SUFFIX}.log"
place_in_cgroup ${MON_PID}

# ---------- vllm vanilla (node0 / GPU 0-3) ----------
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ${WRAP_V} "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    > "${BASE}/logs/vanilla_${SUFFIX}.log" 2>&1 &
V_PID=$!
place_in_cgroup ${V_PID}

# ---------- vllm trident (node1 / GPU 4-7) ----------
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup ${WRAP_T} "${VLLM}" serve "${MODEL}" \
    --port 8002 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
    > "${BASE}/logs/trident_${SUFFIX}.log" 2>&1 &
T_PID=$!
place_in_cgroup ${T_PID}

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

# ---------- router (3-method, sub094) ----------
export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1/completions"
export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1/completions"
export AGSD_MODEL="${MODEL}" AGSD_MODEL_SIZE=qwen_7b
export AGSD_CLASSIFIER_WORKERS=4 AGSD_ROUTER_PORT=8000
cd /tmp && nohup ${WRAP_RTR} "$PY" sub094_router.py > "${BASE}/logs/router_${SUFFIX}.log" 2>&1 &
R_PID=$!
place_in_cgroup ${R_PID}
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

# cleanup cgroup (ON only)
if [ "${MODE}" = "on" ] && [ "${CGROUP_OK}" = "yes" ]; then
    sleep 3
    rmdir "${CG_PATH}" 2>/dev/null || true
fi

# release hugepages (ON only)
if [ "${MODE}" = "on" ] && [ "${HUGEPAGES_OK}" = "yes" ]; then
    echo 0 > /proc/sys/vm/nr_hugepages 2>/dev/null || true
fi

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_186 mode=${MODE} done" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
