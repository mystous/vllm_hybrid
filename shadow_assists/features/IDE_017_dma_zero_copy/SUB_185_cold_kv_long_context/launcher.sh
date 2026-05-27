#!/usr/bin/env bash
# SUB_185 — cold-KV decompress long-context first signal
#
# Long-context workload (sonnet repeat, p50 input ~ 8.7K tokens, max-model-len 8192)
# OFF vs ON. ON mode runs a concurrent CPU dequant firer thread (proxy for
# cold-KV decompress overlap with GPU prefill).
#
# 1-run default. Single vllm instance (vanilla TP=4 on GPU 0-3) on port 8001.
# trident instance NOT used in this SUB (single-instance for clarity of CPU
# overlap signal; AGSD routing 비활성).
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/SUB_185_cold_kv_long_context
mkdir -p "${BASE}/logs" "${BASE}/measurements/off" "${BASE}/measurements/on"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

PROMPTS_FILE=/tmp/sub185_prompts.json

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

MODEL="Qwen/Qwen2.5-32B-Instruct"
MODE="${1:-off}"   # 'off' or 'on'
SUFFIX="${MODE}"
OUT_PREFIX="${BASE}/measurements/${MODE}"

# Core sets — physical cores 0-111, no HT, max 100 workers.
# Vanilla vllm (single TP=4 instance): 0-49 (NUMA 0).
# Firer (ON only): 50-55 (NUMA 0, separate from vllm cores).
# Monitor / bench client: 56-99 (NUMA 1 / spare).
VLLM_CORES="0-49"
FIRER_CORES="50-55"
CLIENT_CORES="56-99"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }
echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_185 mode=${MODE} starting" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ---------- prompts presence ----------
if [ ! -s "${PROMPTS_FILE}" ]; then
    echo "[$(ts)] ERROR: prompts file missing: ${PROMPTS_FILE}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    exit 1
fi

# ---------- monitor ----------
taskset -c "${CLIENT_CORES}" "$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor_${SUFFIX}" --interval 0.5 \
    > "${BASE}/logs/monitor_${SUFFIX}.log" 2>&1 &
MON_PID=$!
echo "[monitor] pid=${MON_PID}" >> "${BASE}/logs/main_${SUFFIX}.log"

# ---------- vllm vanilla (single instance, TP=4, GPU 0-3) ----------
# --max-model-len 8192 covers 8K input + 32 output (8192 >= 8711+32 not strictly,
# but vLLM will reject prompts > model_len; our p50 ~ 8711 > 8192 → some
# prompts will be truncated by vLLM. Bump to 10240 for safety.)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup taskset -c "${VLLM_CORES}" "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 10240 --max-num-seqs 64 --max-num-batched-tokens 32768 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    > "${BASE}/logs/vllm_${SUFFIX}.log" 2>&1 &
V_PID=$!

BOOT_T0=$(date +%s)
echo "[$(ts)] waiting vllm V_PID=${V_PID}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
READY=0
for i in $(seq 1 150); do
    sleep 10
    V_OK=$(curl -sf -m 3 "http://127.0.0.1:8001/v1/models" 2>/dev/null | head -c 30 || echo "")
    if [ -n "$V_OK" ]; then
        READY=1
        break
    fi
done
BOOT_T1=$(date +%s)
BOOT_DUR=$(( BOOT_T1 - BOOT_T0 ))
if [ "${READY}" = "0" ]; then
    echo "[$(ts)] vllm not ready in 25 min, abort" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    kill ${MON_PID} ${V_PID} 2>/dev/null
    pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
    exit 1
fi
echo "[$(ts)] vllm ready ${i}x10s (boot=${BOOT_DUR}s)" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
echo "${BOOT_DUR}" > "${BASE}/logs/boot_${SUFFIX}_seconds.txt"

# ---------- (ON only) start cold-KV CPU dequant firer ----------
FIRER_PID=""
if [ "${MODE}" = "on" ]; then
    echo "[$(ts)] starting cold-KV firer" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    taskset -c "${FIRER_CORES}" "$PY" "${BASE}/src/cold_kv_firer.py" \
        --n-elems 262144 --scale-group 128 --target-hz 100 --duration-s 1800 \
        --out "${BASE}/logs/firer_stats.json" \
        > "${BASE}/logs/firer.log" 2>&1 &
    FIRER_PID=$!
    echo "[firer] pid=${FIRER_PID}" >> "${BASE}/logs/main_${SUFFIX}.log"
    sleep 2
    if ! kill -0 ${FIRER_PID} 2>/dev/null; then
        echo "[$(ts)] WARN: firer did not stay alive; check firer.log" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
    fi
fi

# ---------- benchmark ----------
export BENCH_MODEL="${MODEL}"
OUT="${OUT_PREFIX}/long_sonnet"
mkdir -p "${OUT}"
echo "[$(ts)] === ${MODE} long-context bench 500p x 8K input (max-tokens=32) ===" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
taskset -c "${CLIENT_CORES}" "$PY" "${BASE}/src/bench_long_context.py" \
    --prompts-file "${PROMPTS_FILE}" \
    --url http://127.0.0.1:8001/v1/completions \
    --max-tokens 32 --concurrency 32 \
    --out "${OUT}/bench.json" \
    > "${OUT}/bench.log" 2>&1
RC_BENCH=$?
echo "[$(ts)] bench rc=${RC_BENCH}" | tee -a "${BASE}/logs/main_${SUFFIX}.log"

# ---------- cleanup ----------
echo "[$(ts)] cleanup" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
if [ -n "${FIRER_PID}" ]; then
    kill -TERM ${FIRER_PID} 2>/dev/null
    sleep 2
    kill -9 ${FIRER_PID} 2>/dev/null
fi
sleep 2
pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3
kill ${MON_PID} 2>/dev/null

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')] SUB_185 mode=${MODE} done" | tee -a "${BASE}/logs/main_${SUFFIX}.log"
