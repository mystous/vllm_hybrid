#!/usr/bin/env bash
# [v1.6 Phase 3] cudaEventSync 영역 — FlashAttentionMetadata 영역
# _seq_lens_cpu field 추가. attention.py 의 .cpu() fallback 영역 0 영역.
# baseline Phase 1A 256.8 tps. target +6% → ~272 tps.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="v1.6_phase3_sync_clean"
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

# v1.5 env + Phase 1A workers=4 + Phase 3 backend metadata fix
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
unset VLLM_DEBUG_TORCH_PROFILER

# Phase 1A
export VLLM_NEO_CDEC_WORKERS=4

# Phase 3 검증 — D-cdec-trace 영역 영역 영역 영역 .cpu() fallback 영역 0 영역 검증
export VLLM_DEBUG_CDEC_PATH=1

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[v1.6-phase3] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') starting → ${OUT_DIR}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[v1.6-phase3] launcher PID=${LAUNCHER_PID}"

sleep 270

echo "[v1.6-phase3] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') flamegraph capture"
WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP0/ {print $1; exit}')
[ -z "$WORKER_TP0" ] && WORKER_TP0=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP/ {print $1; exit}')

if [ -n "$WORKER_TP0" ]; then
    timeout 90 "$PYSPY" record -p "$WORKER_TP0" -d 60 \
        --native --idle --threads \
        -f raw -o "${FLAME_DIR}/phase3_raw.txt" 2>&1 &
    wait
fi

sleep 1500

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null
pgrep -f "run_neo_baseline" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[v1.6-phase3] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') DONE"
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
echo "===== D-cdec-trace (.cpu() fallback 영역 영역 영역) ====="
grep "D-cdec-trace" "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo ""
echo "===== 22 항목 fact ====="
echo "SWAP_OUT: $(grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "SWAP_IN:  $(grep -c 'swap-in.*req' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "shape_mm: $(grep -c 'shape mismatch' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "D11_OOB:  $(grep -c 'D11 OOB' "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== crash ====="
echo "assert: $(grep -c 'AssertionError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "cuda:   $(grep -cE 'CUDA error|CUDA-assert' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "segv:   $(grep -cE 'Segfault|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "dead:   $(grep -c 'EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[v1.6-phase3] done"
