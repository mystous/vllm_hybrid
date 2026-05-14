#!/usr/bin/env bash
# [v1.6 smoke test] Phase 1A + Phase 3 통합 영역 정상 동작 영역 영역 검증.
# 5min short — engine init 4min + gen 1min. 22 항목 fire fact + crash 검증.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="v1.6_smoke"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# v1.5 env + Phase 1A workers=4
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
# IDE_006 Phase 2.1 — disable full_mirror so decide_mode produces b1>0
export VLLM_NEO_OPTION_C_FULL_MIRROR=0
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE
unset VLLM_DEBUG_TORCH_PROFILER

# Phase 1A
export VLLM_NEO_CDEC_WORKERS=4
# Phase 3 영역 검증 — .cpu() fallback 영역 0 영역 검증
export VLLM_DEBUG_CDEC_PATH=1

# IDE_006 — NEO per-worker CPU pinning + async cdec mode. OMP=10 same
# as the pin-only winning config; async cdec adds 2 concurrent cdec
# (20 OMP threads on 12 cores = 1.67× oversub, but bounded by pin so
# OS scheduler has work). 16 cores reserved for OS/NCCL/driver.
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
# IDE_006 Phase 1 — NUMA explicit memory bind (KV buffer alloc 시 local NUMA)
export VLLM_NEO_NUMA_BIND=1

echo "[v1.6-smoke] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') starting → ${OUT_DIR}"
echo "[v1.6-smoke] Phase 1A env: VLLM_NEO_CDEC_WORKERS=${VLLM_NEO_CDEC_WORKERS}"
echo "[v1.6-smoke] Phase 3 code: vllm/v1/attention/backends/flash_attn.py _seq_lens_cpu"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 100 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[v1.6-smoke] launcher PID=${LAUNCHER_PID}"

# 5min short (engine init 4min + gen 1min)
sleep 300

# Cleanup
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 $LAUNCHER_PID 2>/dev/null
pgrep -f "run_neo_baseline" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[v1.6-smoke] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') DONE"
echo ""
echo "===== throughput (last 10) ====="
grep -oE "Avg generation throughput: *[0-9.]+" "${LOG_FILE}.stdout" 2>/dev/null | tail -10
echo ""
echo "===== FORK STAT (last) ====="
grep "NEO FORK STAT" "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo ""
echo "===== CDEC_CALL max ====="
grep '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null | grep -oE 'count=[0-9]+' | sort -t= -k2 -n | tail -1
echo ""
echo "===== D-cdec-trace (.cpu() fallback 영역 검증) ====="
grep "D-cdec-trace" "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo ""
echo "===== 22 항목 영역 fact ====="
echo "SWAP_OUT: $(grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "SWAP_IN:  $(grep -c 'swap-in.*req' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "shape_mm: $(grep -c 'shape mismatch' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "D11_OOB:  $(grep -c 'D11 OOB' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "BUF_EXT:  $(grep -c '\[NEO BUF EXTEND\]' "${LOG_FILE}.stdout" 2>/dev/null) / FAIL: $(grep -c '\[NEO BUF EXTEND FAIL\]' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "Opt_I/C/D15+D16: $(grep -c '\[Option I\]' "${LOG_FILE}.stdout" 2>/dev/null) / $(grep -cE '\[Option C / D17C\] first fire|\[Option C / D17C v2' "${LOG_FILE}.stdout" 2>/dev/null) / $(grep -c '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== crash ====="
echo "assert: $(grep -c 'AssertionError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "cuda:   $(grep -cE 'CUDA error|CUDA-assert' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "segv:   $(grep -cE 'Segfault|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "dead:   $(grep -c 'EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "name:   $(grep -c '_os_th.*not defined' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[v1.6-smoke] done"
