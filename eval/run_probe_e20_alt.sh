#!/usr/bin/env bash
# [Phase E20 alternative] gprof rebuild 영역 시간 cost 큼 (100min).
# 기존 build (Release, -O3 -lineinfo) 에서 alternative —
# VLLM_NEO_PROFILE=1 + py-spy --native (C++ frame sample) 동시 실행.
# 결과: NEO component time + C++ pacpu kernel stack.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="E20_alt_neo_profile_pyspy_native"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# v1.5 env
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
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH
unset VLLM_DEBUG_TORCH_PROFILER

# === E20 alt: NEO_PROFILE 활성 ===
export VLLM_NEO_PROFILE=1

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[E20-alt] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

# Launch normal vllm
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 100 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[E20-alt] launcher PID=${LAUNCHER_PID}"

# Wait until engine init + chain firing 활성 (~4분)
sleep 270

# py-spy --native attach to TP0 (60s sample, native C++ frame)
echo "[E20-alt] $(TZ=Asia/Seoul date -Iseconds) py-spy attaching"
WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP0/ {print $1; exit}')
if [ -z "$WORKER_TP0" ]; then
    WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP/ {print $1; exit}')
fi
echo "[E20-alt] target PID=$WORKER_TP0"

if [ -n "$WORKER_TP0" ]; then
    # flamegraph 형식 (visual)
    timeout 90 "$PYSPY" record -p "$WORKER_TP0" -d 60 \
        --native --idle --threads \
        -f flamegraph -o "${OUT_DIR}/pyspy_native_flame.svg" 2>&1 &
    PYSPY_FG_PID=$!
    # raw 형식 (programmatic 분석)
    timeout 90 "$PYSPY" record -p "$WORKER_TP0" -d 60 \
        --native --idle --threads \
        -f raw -o "${OUT_DIR}/pyspy_native_raw.txt" 2>&1 &
    PYSPY_RAW_PID=$!
    wait $PYSPY_FG_PID $PYSPY_RAW_PID 2>/dev/null
    echo "[E20-alt] py-spy done"
fi

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null

echo "[E20-alt] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== NEO PROFILE samples (PER-LAYER ratio) ====="
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | head -2
echo "..."
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | tail -2
echo ""
echo "===== NEO PROFILE avg ratio (cdec / gpu) ====="
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "ratio=[0-9.]+x" | sed 's/ratio=//;s/x//' \
    | awk '{sum+=$1; n++} END {if(n>0) printf "avg=%.2fx n=%d\n", sum/n, n}'
echo ""
echo "===== PROFILE SWAP_OUT / SWAP_IN avg ms ====="
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++} END {if(n>0) printf "SWAP_OUT n=%d avg=%.2fms\n", n, sum/n}'
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++} END {if(n>0) printf "SWAP_IN n=%d avg=%.2fms\n", n, sum/n}'
echo ""
echo "===== py-spy top native C++ frames ====="
if [ -f "${OUT_DIR}/pyspy_native_raw.txt" ]; then
    head -20 "${OUT_DIR}/pyspy_native_raw.txt"
fi
echo ""
echo "===== crash ====="
grep -ciE "died unexpectedly|EngineDeadError" "${LOG_FILE}.stdout" 2>/dev/null
echo "[E20-alt] done"
