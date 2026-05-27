#!/usr/bin/env bash
# SUB_187 — AMX draft head canonical proxy e2e
#
# OFF vs ON modes share a single vanilla vllm instance (Qwen 1.5B, TP=4 GPU0-3)
# - OFF: only vllm + bench client, no AMX firer
# - ON : vllm + concurrent AMX firer at 100 Hz (proxy for cpu_amx draft path
#        of AGSD router); Qwen 0.5B small draft kernel.
#
# Proxy short bench (50 prompts × max-tokens=32 × concurrency=8) for ~30-60 s
# per mode. Total wall ≤ 8 min (boot ~3-5 min + 2× bench ~1 min + cleanup).
set -uo pipefail

BASE=/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter/SUB_187_amx_draft_head
mkdir -p "${BASE}/logs" "${BASE}/measurements/off" "${BASE}/measurements/on"

ROOT=/workspace/vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

# Choose small model for ~3 min boot (vs 10 min Qwen 32B).
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

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

VLLM_CORES="0-49"
FIRER_CORES="50-99"   # 50 cores for AMX (high parallelism)
CLIENT_CORES="50-99"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }
log() { echo "[$(ts)] $*" | tee -a "${BASE}/logs/main.log"; }

log "SUB_187 starting model=${MODEL}"

# Monitor
"$PY" "${ROOT}/eval/monitor.py" "${BASE}/_monitor" --interval 1.0 \
    > "${BASE}/logs/monitor.log" 2>&1 &
MON_PID=$!
log "monitor pid=${MON_PID}"

# vllm vanilla TP=4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup taskset -c "${VLLM_CORES}" "${VLLM}" serve "${MODEL}" \
    --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --gpu-memory-utilization 0.60 \
    --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    > "${BASE}/logs/vllm.log" 2>&1 &
V_PID=$!
log "vllm pid=${V_PID}, waiting up to 15 min"

BOOT_T0=$(date +%s)
READY=0
for i in $(seq 1 90); do
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
    log "vllm not ready in 15 min, abort"
    kill ${MON_PID} ${V_PID} 2>/dev/null
    pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
    exit 1
fi
log "vllm ready after ${BOOT_DUR}s"
echo "${BOOT_DUR}" > "${BASE}/logs/boot_seconds.txt"

# ---------- OFF mode bench (no firer) ----------
log "=== OFF mode bench start ==="
taskset -c "${CLIENT_CORES}" "$PY" "${BASE}/src/proxy_bench.py" \
    --url "http://127.0.0.1:8001/v1/completions" \
    --model "${MODEL}" \
    --max-tokens 32 --concurrency 8 --n-prompts 50 \
    --out "${BASE}/measurements/off/bench.json" \
    > "${BASE}/measurements/off/bench.log" 2>&1
RC_OFF=$?
log "OFF bench rc=${RC_OFF}"

# Quick rest period
sleep 5

# ---------- ON mode bench (with AMX firer concurrent) ----------
log "=== ON mode: starting AMX firer ==="
OMP_NUM_THREADS=50 taskset -c "${FIRER_CORES}" "$PY" "${BASE}/src/amx_firer.py" \
    --target-hz 100 --K 7 --B 1 --duration-s 240 \
    --out "${BASE}/logs/firer_stats.json" \
    > "${BASE}/logs/firer.log" 2>&1 &
F_PID=$!
log "firer pid=${F_PID}"
sleep 3
if ! kill -0 ${F_PID} 2>/dev/null; then
    log "WARN: firer dead, see firer.log"
fi

log "=== ON mode bench start (firer concurrent) ==="
taskset -c "${CLIENT_CORES}" "$PY" "${BASE}/src/proxy_bench.py" \
    --url "http://127.0.0.1:8001/v1/completions" \
    --model "${MODEL}" \
    --max-tokens 32 --concurrency 8 --n-prompts 50 \
    --out "${BASE}/measurements/on/bench.json" \
    > "${BASE}/measurements/on/bench.log" 2>&1
RC_ON=$?
log "ON bench rc=${RC_ON}"

# Cleanup
log "cleanup"
kill -TERM ${F_PID} 2>/dev/null
sleep 2
kill -9 ${F_PID} 2>/dev/null
pgrep -f "vllm serve ${MODEL}" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3
pgrep -f "VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null
kill ${MON_PID} 2>/dev/null

log "SUB_187 done"
