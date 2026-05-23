#!/usr/bin/env bash
# [try94] try86 setup 위에서 CPU 병렬 동적 분석.
# - OMP_NUM_THREADS *미설정* (현재 상태 그대로 — default 224)
# - py-spy native (C++ stack 포함) × 5 dumps
# - top -H -b -d 1 으로 thread 별 CPU usage capture (3 분간)
# - cdec 활성 시점 (4 분 후) 측정 시작
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try94_v5_cpu_profile"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
PYSPY_DIR="${OUT_DIR}/pyspy_native"
mkdir -p "${PYSPY_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# try86 setup 그대로 (v1.4 + Option O2 v2 code + Option O1 v2 env)
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_FORCE_SWAP_IN=0
export VLLM_NEO_SWAP_COOLDOWN=20
export VLLM_NEO_MIRROR_MIN_BUFFER=4
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
unset VLLM_NEO_OPTION_A
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_PROFILE  # per-layer profile 영향 제거

echo "[try94] starting (try86 setup, 동적 CPU 분석) → ${LOG_FILE}"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --num-prompts 500 \
    --target-input-len 8192 \
    --max-tokens 8192 \
    --enable-neo-asymmetric \
    --async-scheduling \
    --enforce-eager false \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" \
    --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[try94] launcher PID=${LAUNCHER_PID}"

# Init + warmup 4 분 (cdec 활성 영역까지)
sleep 240

# Worker PIDs 추출
WORKER_PIDS=$(ps -ef | grep "VLLM::Worker" | grep -v grep | awk '{print $2}' | head -2)
echo "[try94] worker PIDs for analysis: ${WORKER_PIDS}"

# top -H -b -d 1 (3 분간, per-thread CPU usage)
for pid in $WORKER_PIDS; do
    top -H -b -d 1 -n 60 -p "$pid" > "${OUT_DIR}/top_H_w${pid}.txt" 2>&1 &
done

# py-spy native (3 dumps × 30s 간격)
for i in 1 2 3; do
    sleep 30
    for pid in $WORKER_PIDS; do
        DUMP_FILE="${PYSPY_DIR}/native_w${pid}_$(printf '%02d' "$i")_$(date +%H%M%S).txt"
        "$PYSPY" dump --pid "$pid" --native > "$DUMP_FILE" 2>&1 \
            || echo "  native dump failed for $pid"
    done
    echo "[try94] native dump $i done"
done

# 추가 1 분 후 종료
sleep 60
wait 2>/dev/null
kill -TERM $LAUNCHER_PID 2>/dev/null
sleep 3
kill -9 $LAUNCHER_PID 2>/dev/null
ps -ef | grep "EngineCore\|VLLM::Worker" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

echo "[try94] DONE $(date -Iseconds)"

# Quick analysis
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== top -H worker thread count (per snapshot 평균) ====="
for f in "${OUT_DIR}"/top_H_w*.txt; do
    fname=$(basename "$f")
    threads=$(grep -cE "^\s*[0-9]+ root" "$f" 2>/dev/null)
    cpu_total=$(grep -E "^\s*[0-9]+ root" "$f" 2>/dev/null | awk '{sum+=$9} END {print sum}')
    echo "  $fname: thread rows=$threads, sum_CPU%=$cpu_total"
done | head -4
echo ""
echo "===== native py-spy 의 top function (C++ 영역) ====="
for f in "${PYSPY_DIR}"/native_w*_01_*.txt; do
    fname=$(basename "$f")
    echo "--- $fname (top 20 frames) ---"
    head -50 "$f" | head -30
    break  # 첫 dump 만
done
echo "[try94] analysis done"
