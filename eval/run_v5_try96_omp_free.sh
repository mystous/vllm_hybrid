#!/usr/bin/env bash
# [try96] OpenMP 의 affinity 정책 비활성화 — taskset 안에서 thread 자유 활용.
# 변경:
#   OMP_PROC_BIND=false (close → false) — process 의 affinity 변경 안 함
#   OMP_PLACES unset — places 무영향
#   OMP_NUM_THREADS=14 유지
# 측정:
#   taskset -pc 로 worker affinity 직접 확인
#   OMP_DISPLAY_ENV 의 places 변화
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try96_v5_omp_free"
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

# OpenMP — affinity 정책 비활성
export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES
export OMP_DISPLAY_ENV=TRUE

echo "[try96] starting (OMP_PROC_BIND=false, OMP_PLACES unset, taskset 0-111) → ${LOG_FILE}"
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
echo "[try96] launcher PID=${LAUNCHER_PID}"

# Init 4 분
sleep 240

# Worker affinity 직접 측정
echo "===== Worker CPU affinity (taskset -pc) =====" >> "${OUT_DIR}/affinity.txt"
for pid in $(ps -ef | grep "VLLM::Worker" | grep -v grep | awk '{print $2}'); do
    affinity=$(taskset -pc "$pid" 2>&1)
    echo "$affinity" >> "${OUT_DIR}/affinity.txt"
done
cat "${OUT_DIR}/affinity.txt"

# top -H + py-spy
WORKER_PIDS=$(ps -ef | grep "VLLM::Worker" | grep -v grep | awk '{print $2}' | head -2)
for pid in $WORKER_PIDS; do
    top -H -b -d 1 -n 60 -p "$pid" > "${OUT_DIR}/top_H_w${pid}.txt" 2>&1 &
done

for i in 1 2 3; do
    sleep 30
    for pid in $WORKER_PIDS; do
        "$PYSPY" dump --pid "$pid" --native > "${PYSPY_DIR}/native_w${pid}_$(printf '%02d' "$i").txt" 2>&1 \
            || echo "  dump fail $pid"
    done
done

# 1 분 buffer + 종료
sleep 30
wait 2>/dev/null

# 강제 kill — process tree 다 잡아냄
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null
echo "[try96] DONE $(date -Iseconds)"

echo ""
echo "===== Worker affinity ====="
cat "${OUT_DIR}/affinity.txt" 2>/dev/null
echo ""
echo "===== OMP_DISPLAY_ENV (worker) ====="
grep -E '^  OMP_(NUM|PLACES|PROC_BIND)' "${LOG_FILE}.stdout" 2>/dev/null | head -20
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo ""
echo "===== [OMP DBG] log first 5 + last 5 ====="
grep '\[OMP DBG\]' "${LOG_FILE}.stdout" 2>/dev/null | head -5
echo "---last 5---"
grep '\[OMP DBG\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo "[try96] analysis done"
