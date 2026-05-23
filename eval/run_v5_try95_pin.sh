#!/usr/bin/env bash
# [try95] HT sibling pinning 회피 + OMP env tune.
# - taskset -c 0-111 : 모든 process 가 physical core 0-111 만 사용
#   HT sibling (112-223) 회피 — 시스템 SMT 변경 X
# - OMP env: per-worker 14 threads (8 workers × 14 = 112 cores) physical bind
# - pacpu kernel 에 [OMP DBG] log 추가 — runtime ws / num_threads 측정
# - 동적 분석: top -H + py-spy native
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try95_v5_pin"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
PYSPY_DIR="${OUT_DIR}/pyspy_native"
mkdir -p "${PYSPY_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# try86 setup
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
unset VLLM_NEO_PROFILE

# OpenMP env — per-worker 14 threads (8 worker × 14 = 112 physical cores).
# OMP_PLACES=cores : thread 가 physical core 에 bind (HT sibling 회피)
# OMP_PROC_BIND=close : thread 가 master 의 close core 에 stay (NUMA-local)
export OMP_NUM_THREADS=14
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=TRUE

echo "[try95] starting (taskset 0-111 + OMP per-worker 14 threads) → ${LOG_FILE}"
# taskset 으로 전체 process tree 가 cpu 0-111 만 사용 (HT sibling 회피).
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
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
echo "[try95] launcher PID=${LAUNCHER_PID}"

# Init + warmup 4 분
sleep 240

# Worker PIDs
WORKER_PIDS=$(ps -ef | grep "VLLM::Worker" | grep -v grep | awk '{print $2}' | head -2)
echo "[try95] worker PIDs for analysis: ${WORKER_PIDS}"

# top -H 3 분간
for pid in $WORKER_PIDS; do
    top -H -b -d 1 -n 90 -p "$pid" > "${OUT_DIR}/top_H_w${pid}.txt" 2>&1 &
done

# py-spy native 5 dumps × 30s
for i in 1 2 3 4 5; do
    sleep 30
    for pid in $WORKER_PIDS; do
        DUMP_FILE="${PYSPY_DIR}/native_w${pid}_$(printf '%02d' "$i").txt"
        "$PYSPY" dump --pid "$pid" --native > "$DUMP_FILE" 2>&1 \
            || echo "  dump fail $pid"
    done
done

# 추가 1 분
sleep 30
wait 2>/dev/null
kill -TERM $LAUNCHER_PID 2>/dev/null
sleep 3
kill -9 $LAUNCHER_PID 2>/dev/null
ps -ef | grep "EngineCore\|VLLM::Worker" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

echo "[try95] DONE $(date -Iseconds)"

# Analysis
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== OMP DBG log (first 10) ====="
grep '\[OMP DBG\]' "${LOG_FILE}.stdout" 2>/dev/null | head -10
echo ""
echo "===== OMP DBG log (last 5) ====="
grep '\[OMP DBG\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo ""
echo "===== OMP_DISPLAY_ENV output ====="
grep -E 'OMP_(NUM_THREADS|PLACES|PROC_BIND)' "${LOG_FILE}.stdout" 2>/dev/null | head -10
echo "[try95] analysis done"
