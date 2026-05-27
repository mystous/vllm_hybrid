#!/usr/bin/env bash
# SUB_194 — multi-run variance verification for Top-3 net positive levers
#
# Levers under test:
#   L183 = SUB_183 NUMA pin (env wrapper: numactl --membind --cpunodebind + taskset)
#   L188 = SUB_188 softmax precompute (side-channel binary, 100ms cycle, cores 80-95)
#   L190 = SUB_190 async tokenizer worker (side-channel binary, 20ms cycle, cores 80-95)
#
# Per (lever, mode) cycle: 1 fresh vllm boot, then 3 back-to-back agsd-gated 500p×3-mix runs.
# Total cycles = 3 levers × 2 modes (off, on) × 1 boot = 6 vllm boots.
# Per cycle wall ≈ 80s boot + 3 × (3 mix × ~30s) ≈ 350s ≈ 6 min. Total ≈ 35 min.
#
# Only AGSD-gated scenario is measured (vanilla-only / trident-only individual cells skipped).
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_015_cpu_extreme_util/SUB_194_multi_run_variance
mkdir -p "${BASE}/logs" "${BASE}/measurements"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

SOFTMAX_BIN=/workspace/vllm_hybrid/shadow_assists/features/IDE_018_phase_burst/SUB_188_side_channel_precompute/build/side_channel_precompute
TOK_BIN=/workspace/vllm_hybrid/shadow_assists/features/IDE_016_avx512_amx_pool/SUB_190_async_tokenizer_worker/build/async_tokenizer_worker

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

LEVER="${1:-L183}"   # L183 / L188 / L190
MODE="${2:-off}"     # off / on
N_RUNS="${3:-3}"
SUFFIX="${LEVER}_${MODE}"
OUT_BASE="${BASE}/measurements/${LEVER}/${MODE}"
LOG="${BASE}/logs/main_${SUFFIX}.log"
mkdir -p "${OUT_BASE}"

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
unset VLLM_USE_PHASE_BURST VLLM_PHASE_BURST_DUMMY_FILL || true

MODEL="Qwen/Qwen2.5-32B-Instruct"

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_194 lever=${LEVER} mode=${MODE} N=${N_RUNS} starting" | tee -a "${LOG}"

# ── Determine wrappers (NUMA pin only for L183 ON) ─────────────────
NUMACTL_V=""
NUMACTL_T=""
if [ "${LEVER}" = "L183" ] && [ "${MODE}" = "on" ]; then
    NUMACTL_V="numactl --membind=0 --cpunodebind=0"
    NUMACTL_T="numactl --membind=1 --cpunodebind=1"
    echo "[$(ts)] NUMA pin applied (vanilla=node0, trident=node1)" | tee -a "${LOG}"
fi

# ── monitor ────────────────────────────────────────────────────────
"$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor_${SUFFIX}" --interval 0.5 > "${BASE}/logs/monitor_${SUFFIX}.log" 2>&1 &
MON_PID=$!
echo "[monitor] pid=${MON_PID}" >> "${LOG}"

# ── vllm vanilla (GPU 0-3) ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ${NUMACTL_V} "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    > "${BASE}/logs/vanilla_${SUFFIX}.log" 2>&1 &
V_PID=$!

# ── vllm trident (GPU 4-7) ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup ${NUMACTL_T} "${VLLM}" serve "${MODEL}" \
    --port 8002 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
    > "${BASE}/logs/trident_${SUFFIX}.log" 2>&1 &
T_PID=$!

BOOT_T0=$(date +%s)
echo "[$(ts)] waiting vllm (V_PID=${V_PID} T_PID=${T_PID})" | tee -a "${LOG}"
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
    echo "[$(ts)] vllm not ready in 20 min, abort" | tee -a "${LOG}"
    kill ${MON_PID} ${V_PID} ${T_PID} 2>/dev/null
    pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
    exit 1
fi
echo "[$(ts)] vllm ready (boot=${BOOT_DUR}s)" | tee -a "${LOG}"
echo "${BOOT_DUR}" > "${BASE}/logs/boot_${SUFFIX}_seconds.txt"

# ── side-channel binary (L188 / L190 ON only) ──────────────────────
SC_PID=""
if [ "${MODE}" = "on" ]; then
    SC_BIN=""
    case "${LEVER}" in
        L188) SC_BIN="${SOFTMAX_BIN}" ;;
        L190) SC_BIN="${TOK_BIN}" ;;
        L183) SC_BIN="" ;;  # NUMA = no extra binary
    esac
    if [ -n "${SC_BIN}" ]; then
        if [ ! -x "${SC_BIN}" ]; then
            echo "[$(ts)] side-channel binary missing: ${SC_BIN}" | tee -a "${LOG}"
        else
            nohup "${SC_BIN}" > "${BASE}/logs/sidechannel_${SUFFIX}.log" 2>&1 &
            SC_PID=$!
            echo "[$(ts)] side-channel pid=${SC_PID} bin=${SC_BIN}" | tee -a "${LOG}"
            sleep 2
            if ! kill -0 ${SC_PID} 2>/dev/null; then
                echo "[$(ts)] side-channel failed to start" | tee -a "${LOG}"
            fi
        fi
    fi
fi

# ── router ─────────────────────────────────────────────────────────
export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1/completions"
export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1/completions"
export AGSD_MODEL="${MODEL}" AGSD_MODEL_SIZE=qwen_7b
export AGSD_CLASSIFIER_WORKERS=4 AGSD_ROUTER_PORT=8000
cd /tmp && nohup "$PY" sub094_router.py > "${BASE}/logs/router_${SUFFIX}.log" 2>&1 &
R_PID=$!
sleep 10

H=$(curl -sf -m 3 "http://127.0.0.1:8000/health" 2>/dev/null || echo "")
echo "[$(ts)] router health: ${H}" | tee -a "${LOG}"

# ── multi-run benchmark loop (agsd-gated only) ─────────────────────
export BENCH_MODEL="${MODEL}"
for RUN in $(seq 1 ${N_RUNS}); do
    echo "[$(ts)] ===== run ${RUN}/${N_RUNS} (agsd-only, 500p × 3 mix) =====" | tee -a "${LOG}"
    for MIX in balanced sonnet-heavy code-heavy; do
        OUT="${OUT_BASE}/run${RUN}/${MIX}"
        mkdir -p "${OUT}"
        curl -sX POST http://127.0.0.1:8000/reset > /dev/null
        echo "[$(ts)] run=${RUN} mix=${MIX}" | tee -a "${LOG}"
        "$PY" "${BASE}/bench_agsd_only.py" \
            --num-prompts 500 --max-tokens 32 --concurrency 32 \
            --mix ${MIX} --out-dir "${OUT}/" \
            > "${OUT}/bench.log" 2>&1
        curl -sf -m 3 "http://127.0.0.1:8000/stats" > "${OUT}/router_stats.json" 2>/dev/null || true
    done
done

# ── side-channel stats ─────────────────────────────────────────────
if [ -n "${SC_PID}" ]; then
    echo "[$(ts)] stopping side-channel pid=${SC_PID}" | tee -a "${LOG}"
    kill -TERM ${SC_PID} 2>/dev/null
    sleep 2
    kill -9 ${SC_PID} 2>/dev/null
    tail -5 "${BASE}/logs/sidechannel_${SUFFIX}.log" 2>/dev/null | tee -a "${LOG}"
fi

# ── cleanup ────────────────────────────────────────────────────────
echo "[$(ts)] cleanup" | tee -a "${LOG}"
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

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_194 lever=${LEVER} mode=${MODE} done" | tee -a "${LOG}"
