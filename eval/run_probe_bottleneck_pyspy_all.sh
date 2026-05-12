#!/usr/bin/env bash
# [Run-A] py-spy 9 프로세스 동시 — 22항목 Flamegraph 기반 검증.
# Worker_TP0..7 (8개) + EngineCore (1개) 동시 샘플.
# --native --idle --threads: C++ frame + idle thread + thread pool 모두 캡처.
# 산출물: raw_tp{0..7}.txt, raw_enginecore.txt, STACK_CHECK 표
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="RunA_pyspy_all_22items"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
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
export VLLM_NEO_OPTION_C_FULL_MIRROR=0  # Phase 2 routing fix
export VLLM_NEO_CDEC_WORKERS=4
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE VLLM_DEBUG_CDEC_PATH
unset VLLM_DEBUG_TORCH_PROFILER

export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
# IDE_006 winning config — per-worker CPU pinning
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
# Phase 1 — NUMA bind
export VLLM_NEO_NUMA_BIND=1

echo "[Run-A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') starting → ${OUT_DIR}"
echo "[Run-A] 타겟: Worker_TP0..7 (8개) + EngineCore (1개)"

# Engine 실행 (500p, chain firing 안정화)
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[Run-A] launcher PID=${LAUNCHER_PID}"

# Engine init 대기 (chain firing 안정화 ~270s)
echo "[Run-A] engine init 대기 270s..."
sleep 270
echo "[Run-A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') py-spy 시작"

# === 9 프로세스 PID 탐지 ===
# Worker_TP0..7: comm = VLLM::Worker_TP (15자 truncation → VLLM::Worker_TP 매칭)
mapfile -t WORKER_PIDS < <(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::Worker_TP/ {print $1}')
# EngineCore: comm = VLLM::EngineCor (truncated)
EC_PID=$(ps -o pid,comm -A 2>/dev/null | awk '/VLLM::EngineCor/ {print $1; exit}')

echo "[Run-A] Worker PIDs: ${WORKER_PIDS[*]:-none}"
echo "[Run-A] EngineCore PID: ${EC_PID:-none}"

PYSPY_PIDS=()

# Worker 각각 py-spy
for pid in "${WORKER_PIDS[@]}"; do
    # TP 번호 추출 (comm에서 마지막 숫자, 없으면 PID 사용)
    COMM=$(ps -p "$pid" -o comm= 2>/dev/null || echo "")
    N=$(echo "$COMM" | grep -oE '[0-9]+$' || echo "$pid")
    echo "[Run-A]   py-spy worker PID=$pid comm=$COMM → raw_tp${N}.txt"
    timeout 120 "$PYSPY" record \
        -p "$pid" -d 60 -r 100 \
        --native --idle --threads \
        -f raw -o "${OUT_DIR}/raw_tp${N}.txt" 2>"${OUT_DIR}/pyspy_tp${N}.err" &
    PYSPY_PIDS+=($!)
done

# EngineCore py-spy
if [ -n "${EC_PID:-}" ]; then
    echo "[Run-A]   py-spy EngineCore PID=$EC_PID → raw_enginecore.txt"
    timeout 120 "$PYSPY" record \
        -p "$EC_PID" -d 60 -r 100 \
        --native --idle --threads \
        -f raw -o "${OUT_DIR}/raw_enginecore.txt" 2>"${OUT_DIR}/pyspy_enginecore.err" &
    PYSPY_PIDS+=($!)
else
    echo "[Run-A] WARNING: EngineCore PID 탐지 실패 — 스케줄러 항목 미샘플"
fi

echo "[Run-A] py-spy ${#PYSPY_PIDS[@]}개 프로세스 백그라운드 실행 중..."
wait "${PYSPY_PIDS[@]}" 2>/dev/null
echo "[Run-A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') py-spy 완료"

# === Cleanup ===
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
ps -o pid,comm -A 2>/dev/null | awk '/VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null
kill -9 "$LAUNCHER_PID" 2>/dev/null
pgrep -f "run_neo_baseline" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo ""
echo "===== 수집된 raw 파일 ====="
ls -lh "${OUT_DIR}"/raw_*.txt 2>/dev/null || echo "없음"

echo ""
echo "===== 22항목 STACK_CHECK (Flamegraph 기반 검증) ====="
echo "--- Worker 프로세스 (항목 2,3,4,7,9,10,12,13,14,16,21,22) ---"
for kw in \
    "_neo_cdec_compute_cpu" \
    "forward_double" \
    "forward_neo_pipelined" \
    "unified_attention_with_output" \
    "ispc_attention_tasks" \
    "forward_pipeline" \
    "_neo_handle_kv_swap" \
    "ensure_capacity" \
    "neo_swap_in_alloc"; do
    cnt=$(cat "${OUT_DIR}"/raw_tp*.txt 2>/dev/null | grep -c "$kw" || echo 0)
    echo "  STACK_CHECK [Worker] [$kw]: $cnt"
done

echo "--- EngineCore 프로세스 (항목 1,5,6,8,11,20) ---"
for kw in \
    "_handle_neo_swaps" \
    "NeoScheduler" \
    "NeoSchedulerAdapter"; do
    cnt=$(cat "${OUT_DIR}"/raw_enginecore.txt 2>/dev/null | grep -c "$kw" || echo 0)
    echo "  STACK_CHECK [EngineCore] [$kw]: $cnt"
done

echo ""
echo "===== 22항목 결과 요약 ====="
# 항목별 판정 (keyword → 항목번호 매핑)
declare -A KW_ITEMS=(
    ["_handle_neo_swaps"]="#1,#8"
    ["_neo_cdec_compute_cpu"]="#2,#10,#16"
    ["forward_double"]="#3"
    ["forward_neo_pipelined"]="#4"
    ["NeoScheduler"]="#5"
    ["NeoSchedulerAdapter"]="#6,#11,#20"
    ["unified_attention_with_output"]="#7"
    ["ispc_attention_tasks"]="#9"
    ["forward_pipeline"]="#13"
    ["_neo_handle_kv_swap"]="#12,#14"
    ["ensure_capacity"]="#21"
    ["neo_swap_in_alloc"]="#22"
)
for kw in "${!KW_ITEMS[@]}"; do
    if [ -f "${OUT_DIR}/raw_enginecore.txt" ] && grep -q "$kw" "${OUT_DIR}/raw_enginecore.txt" 2>/dev/null; then
        echo "  ✓ ${KW_ITEMS[$kw]} [$kw] — EngineCore"
    elif cat "${OUT_DIR}"/raw_tp*.txt 2>/dev/null | grep -q "$kw"; then
        echo "  ✓ ${KW_ITEMS[$kw]} [$kw] — Worker"
    else
        echo "  ✗ ${KW_ITEMS[$kw]} [$kw] — 미검출 ← 추가 추적 필요"
    fi
done

echo ""
echo "===== throughput (last 5) ====="
grep -oE "Avg generation throughput: *[0-9.]+" "${LOG_FILE}.stdout" 2>/dev/null | tail -5

echo ""
echo "===== FORK STAT (last) ====="
grep "NEO FORK STAT" "${LOG_FILE}.stdout" 2>/dev/null | tail -1

echo ""
echo "===== crash check ====="
echo "assert: $(grep -c 'AssertionError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "cuda:   $(grep -cE 'CUDA error|CUDA-assert' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "segv:   $(grep -cE 'Segfault|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "dead:   $(grep -c 'EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"

echo ""
echo "[Run-A] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') DONE → ${OUT_DIR}"
