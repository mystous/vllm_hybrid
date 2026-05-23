#!/usr/bin/env bash
# [v1.6 Phase 1A] cdec executor max_workers 영역 4 — env 변경만.
# baseline (Phase 0): 214 tps. target: 268 tps (+25%).
# 30min sustain + flamegraph + 22 항목 monitor.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="v1.6_phase1A_workers4"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"
FLAME_DIR="${OUT_DIR}/flamegraph"
mkdir -p "${FLAME_DIR}"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# v1.5 env + Phase 1A fix
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
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE
unset VLLM_DEBUG_CDEC_PATH VLLM_DEBUG_TORCH_PROFILER

# === Phase 1A C1A-1 fix ===
export VLLM_NEO_CDEC_WORKERS=4

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[v1.6-phase1A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') starting → ${OUT_DIR}"
echo "[v1.6-phase1A] VLLM_NEO_CDEC_WORKERS=${VLLM_NEO_CDEC_WORKERS}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[v1.6-phase1A] launcher PID=${LAUNCHER_PID}"

sleep 270

# flamegraph capture
echo "[v1.6-phase1A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') flamegraph capture"
WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP0/ {print $1; exit}')
if [ -z "$WORKER_TP0" ]; then
    WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP/ {print $1; exit}')
fi
echo "[v1.6-phase1A] target PID=$WORKER_TP0"

if [ -n "$WORKER_TP0" ]; then
    timeout 90 "$PYSPY" record -p "$WORKER_TP0" -d 60 \
        --native --idle --threads \
        -f raw -o "${FLAME_DIR}/phase1A_raw.txt" 2>&1 &
    PYSPY_RAW_PID=$!
    wait $PYSPY_RAW_PID 2>/dev/null
    echo "[v1.6-phase1A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') flamegraph done"
fi

# 30min full run 잔여 (4분 + 60s = 5분 → 25분 추가)
sleep 1500

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null
pgrep -f "run_neo_baseline" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[v1.6-phase1A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') DONE"
echo ""
echo "===== throughput last 50 avg ====="
grep -oE "Avg generation throughput: *[0-9.]+" "${LOG_FILE}.stdout" 2>/dev/null \
    | tail -50 | grep -oE "[0-9]+\.[0-9]+" \
    | awk '{sum+=$1; n++} END {if(n>0) printf "avg=%.1f tps n=%d\n", sum/n, n}'
echo ""
echo "===== FORK STAT (last) ====="
grep "NEO FORK STAT" "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo ""
echo "===== CDEC_CALL max ====="
grep '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null | grep -oE 'count=[0-9]+' | sort -t= -k2 -n | tail -1
echo ""
echo "===== crash ====="
echo "assert: $(grep -c 'AssertionError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "cuda:   $(grep -cE 'CUDA error|CUDA-assert' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "segv:   $(grep -cE 'Segfault|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "dead:   $(grep -c 'EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== flamegraph compare (Phase 0 baseline vs Phase 1A) ====="
BASELINE_RAW="${ROOT_DIR}/eval/results/20260512_121615_v1.6_phase0_baseline/flamegraph/baseline_raw.txt"
if [ -f "$BASELINE_RAW" ] && [ -f "${FLAME_DIR}/phase1A_raw.txt" ]; then
    echo "[baseline] pthread_mutex_lock count:"
    grep -c "pthread_mutex_lock" "$BASELINE_RAW"
    echo "[phase1A] pthread_mutex_lock count:"
    grep -c "pthread_mutex_lock" "${FLAME_DIR}/phase1A_raw.txt"
    echo ""
    echo "[baseline] pthread_mutex_unlock count:"
    grep -c "pthread_mutex_unlock" "$BASELINE_RAW"
    echo "[phase1A] pthread_mutex_unlock count:"
    grep -c "pthread_mutex_unlock" "${FLAME_DIR}/phase1A_raw.txt"
fi
echo "[v1.6-phase1A] done"
