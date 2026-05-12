#!/usr/bin/env bash
# [Phase E17] cuda-gdb — running worker process attach + thread state snapshot.
# silent SEGV root 가 NameError 로 식별돼 core dump 분석은 不要.
# 현 동작 중인 process 의 GPU + CPU thread state snapshot 만 추출.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="E17_cuda_gdb"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_MIRROR_MIN_BUFFER=8
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
export VLLM_NEO_OPTION_C_FULL_MIRROR=1
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE VLLM_DEBUG_CDEC_PATH

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[E17-cuda-gdb] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

# Launch normal 5min run
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 100 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[E17-cuda-gdb] launcher PID=${LAUNCHER_PID}"

# Wait until chain firing 활성 (engine init ~4min)
sleep 270

echo "[E17-cuda-gdb] $(TZ=Asia/Seoul date -Iseconds) attaching cuda-gdb to TP0"
WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP0/ {print $1; exit}')
if [ -z "$WORKER_TP0" ]; then
    WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker/ {print $1; exit}')
fi

if [ -n "$WORKER_TP0" ]; then
    echo "[E17-cuda-gdb] attaching to PID=$WORKER_TP0"
    timeout 60 /usr/local/cuda/bin/cuda-gdb \
        -batch \
        -ex "set pagination off" \
        -ex "set print thread-events off" \
        -ex "attach $WORKER_TP0" \
        -ex "thread apply all bt" \
        -ex "info cuda kernels" \
        -ex "info cuda threads" \
        -ex "detach" \
        -ex "quit" \
        > "${OUT_DIR}/cuda_gdb_TP0.log" 2>&1
    echo "[E17-cuda-gdb] attached + stack dumped"
else
    echo "[E17-cuda-gdb] worker not found"
fi

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null

echo "[E17-cuda-gdb] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== cuda-gdb thread bt 영역 ====="
head -100 "${OUT_DIR}/cuda_gdb_TP0.log" 2>/dev/null
echo ""
echo "===== cuda kernels (active GPU work) ====="
grep -A 5 "info cuda kernels" "${OUT_DIR}/cuda_gdb_TP0.log" 2>/dev/null | head -20
echo "[E17-cuda-gdb] done"
