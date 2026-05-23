#!/usr/bin/env bash
# [Phase 6.2 / 6P] NEO PROFILE — cdec dispatch + swap_in/out component time.
# 기존 코드 적재된 영역 (attention.py:1030-1116, gpu_model_runner.py:6406-6510)
# 의 VLLM_NEO_PROFILE=1 env 만 활성. 5분 short measurement.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="neo_profile_component"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === try105 env (chain firing 95.6%) ===
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
# IDE_006 Phase 4.2 — NEO 정통 Step 2/3 inline 활성. NeoScheduler 의
# step_2_3_only() 호출 (perfpredictor hot-path 회피). NEO 원본 정통
# threshold (95-100%) + hysteresis 동작. ratio 1.0 = NEO paper spec.
# 본 환경 (H100×8) 에서 KV 100% 안 닿음 → ratio 작게 해야 fire.
export VLLM_NEO_NEOSCHED_STEP23=1
export VLLM_NEO_SWAP_OUT_RATIO=0.5
# 6-step driving — dry-run observe mode (queue save/restore, no mutation).
# DRY_RUN=0 시 apply mode (swap_out/swap_in 실제 적용).
export VLLM_NEO_DRIVE_6STEP=1
export VLLM_NEO_6STEP_DRY_RUN=1
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_MIRROR_MIN_BUFFER=8
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
# IDE_006 Phase 2.1 — disable full_mirror brute-force so decide_mode
# (mode_selector.decide_mode) is called and produces balanced sub_batches
# (b1 > 0). This restores NEO §4.4 asymmetric pipeline intent.
export VLLM_NEO_OPTION_C_FULL_MIRROR=0
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_FUSED_RMSNORM
unset VLLM_NEO_DISABLE_SWAP_IN VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN VLLM_NEO_SWAP_COOLDOWN
unset VLLM_DEBUG_FAULTHANDLER

# === VLLM_NEO_PROFILE 활성 (분석 단계 한정) ===
export VLLM_NEO_PROFILE=1

export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
# IDE_006 winning config — per-worker CPU pinning (pin=12, OMP=10)
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
# IDE_006 Phase 1 — NUMA explicit memory bind
export VLLM_NEO_NUMA_BIND=1
# IDE_006 Phase 3 — async cdec + deeper pipeline (default OFF — 본
# 환경에서 OMP 경쟁으로 회귀 -62%. 코드는 보존, 필요 시 env 로 enable).
# export VLLM_NEO_ASYNC_CDEC=1
# export VLLM_NEO_CDEC_PIPELINE_DEPTH=2

echo "[neo_profile] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"
echo "[neo_profile] VLLM_NEO_PROFILE=${VLLM_NEO_PROFILE}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[neo_profile] launcher PID=${LAUNCHER_PID}"

# 8분 measurement — dry-run 6-step driving deadlock 회피 + observe
sleep 480

# Cleanup
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[neo_profile] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -5
echo ""
echo "===== PROFILE PER-LAYER (attention.py — cdec wait + GPU forward component time) ====="
grep "PROFILE PER-LAYER" "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo ""
echo "===== PROFILE SWAP_OUT (gpu_model_runner.py — swap_out per-call ms) ====="
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "..."
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "total SWAP_OUT events: $(grep -c "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== PROFILE SWAP_IN (gpu_model_runner.py — swap_in per-call ms) ====="
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "..."
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "total SWAP_IN events: $(grep -c "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null)"
echo ""
echo "===== SWAP_OUT elapsed_ms 통계 ====="
grep "PROFILE SWAP_OUT" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++; if($1>max||NR==1)max=$1; if($1<min||NR==1)min=$1} END {if(n>0) printf "n=%d avg=%.2f min=%.2f max=%.2f ms\n", n, sum/n, min, max}'
echo ""
echo "===== SWAP_IN elapsed_ms 통계 ====="
grep "PROFILE SWAP_IN" "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//' \
    | awk '{sum+=$1; n++; if($1>max||NR==1)max=$1; if($1<min||NR==1)min=$1} END {if(n>0) printf "n=%d avg=%.2f min=%.2f max=%.2f ms\n", n, sum/n, min, max}'
echo ""
echo "===== crash check ====="
echo "crash: $(grep -ciE 'died unexpectedly|EngineDeadError' "${LOG_FILE}.stdout" 2>/dev/null)"
echo "[neo_profile] analysis done"
