#!/usr/bin/env bash
# [Run-H] 동일 조건 baseline — vanilla vs current 비교.
# 방법론 수정: wall_s + full output_tps (last-N 아님) + 동일 시드/조건.
# result.json 필드: output_tps, generate_wall_s, init_s
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="RunH_baseline_comparison"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# 공통 실행 파라미터 (동일 조건)
NUM_PROMPTS=500
SEED=0
MAX_TOKENS=8192
TARGET_INPUT=8192
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=256
GPU_UTIL=0.85
TP=8
BATCHED_TOKENS=8192

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

# ==============================================================
# === RUN 1: VANILLA (NEO 없음) ===
# ==============================================================
echo ""
echo "========================================================"
echo "[Run-H] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') === VANILLA RUN 시작 ==="
echo "========================================================"

# NEO 관련 env 모두 제거
unset VLLM_NEO_PREDICTOR VLLM_NEO_LOAD_AWARE_MIN_RUNNING \
      VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP VLLM_NEO_FORCE_SWAP_IN \
      VLLM_NEO_MAX_SWAP_IN_PER_STEP VLLM_NEO_CPU_RESIDENT_REQS \
      VLLM_NEO_SWAP_IN_ORDER VLLM_NEO_MIRROR_MIN_BUFFER \
      VLLM_NEO_OPTION_K VLLM_NEO_OPTION_C VLLM_NEO_OPTION_L \
      VLLM_NEO_OPTION_M2 VLLM_NEO_OPTION_C_FULL_MIRROR \
      VLLM_NEO_CDEC_WORKERS VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A \
      VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE VLLM_DEBUG_CDEC_PATH \
      VLLM_DEBUG_TORCH_PROFILER 2>/dev/null || true

VANILLA_LOG="${OUT_DIR}/vanilla.log"
VANILLA_JSON="${OUT_DIR}/vanilla_result.json"

VANILLA_START=$(date +%s)
VANILLA_START_KST=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')
echo "[Run-H] vanilla 시작: ${VANILLA_START_KST}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size ${TP} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --num-prompts ${NUM_PROMPTS} \
    --target-input-len ${TARGET_INPUT} \
    --max-tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    --async-scheduling \
    --enforce-eager false \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens ${BATCHED_TOKENS} \
    --log-file "${VANILLA_LOG}" \
    --output-file "${VANILLA_JSON}" \
    > "${VANILLA_LOG}.stdout" 2>&1
VANILLA_RC=$?
VANILLA_END=$(date +%s)
VANILLA_WALL=$((VANILLA_END - VANILLA_START))
echo "[Run-H] vanilla 종료: $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') (wall=${VANILLA_WALL}s, rc=${VANILLA_RC})"

# GPU 완전 해제 대기
sleep 10

# ==============================================================
# === RUN 2: CURRENT (NEO + Phase 1A + Phase 3) ===
# ==============================================================
echo ""
echo "========================================================"
echo "[Run-H] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') === CURRENT (NEO) RUN 시작 ==="
echo "========================================================"

# v1.5 env + Phase 1A workers=4 + Phase 3 (flash_attn.py 변경은 코드에 반영됨)
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
export VLLM_NEO_CDEC_WORKERS=4
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A
unset VLLM_DEBUG_FAULTHANDLER VLLM_NEO_PROFILE VLLM_DEBUG_CDEC_PATH
unset VLLM_DEBUG_TORCH_PROFILER

CURRENT_LOG="${OUT_DIR}/current.log"
CURRENT_JSON="${OUT_DIR}/current_result.json"

CURRENT_START=$(date +%s)
CURRENT_START_KST=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')
echo "[Run-H] current 시작: ${CURRENT_START_KST}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size ${TP} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --num-prompts ${NUM_PROMPTS} \
    --target-input-len ${TARGET_INPUT} \
    --max-tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    --enable-neo-asymmetric \
    --async-scheduling \
    --enforce-eager false \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens ${BATCHED_TOKENS} \
    --log-file "${CURRENT_LOG}" \
    --output-file "${CURRENT_JSON}" \
    > "${CURRENT_LOG}.stdout" 2>&1
CURRENT_RC=$?
CURRENT_END=$(date +%s)
CURRENT_WALL=$((CURRENT_END - CURRENT_START))
echo "[Run-H] current 종료: $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') (wall=${CURRENT_WALL}s, rc=${CURRENT_RC})"

# ==============================================================
# === 비교 결과 출력 ===
# ==============================================================
echo ""
echo "========================================================"
echo "[Run-H] 비교 결과"
echo "========================================================"

parse_json() {
    local f="$1"
    if [ -f "$f" ]; then
        python3 -c "
import json, sys
d = json.load(open('$f'))
print(f\"output_tps={d.get('output_tps', 'N/A'):.2f}\" if isinstance(d.get('output_tps'), (int,float)) else f\"output_tps={d.get('output_tps','N/A')}\")
print(f\"generate_wall_s={d.get('generate_wall_s', 'N/A'):.1f}\" if isinstance(d.get('generate_wall_s'), (int,float)) else f\"generate_wall_s={d.get('generate_wall_s','N/A')}\")
print(f\"init_s={d.get('init_s', 'N/A'):.1f}\" if isinstance(d.get('init_s'), (int,float)) else f\"init_s={d.get('init_s','N/A')}\")
print(f\"total_output_tokens={d.get('total_output_tokens','N/A')}\")
" 2>/dev/null || echo "파싱 실패"
    else
        echo "result.json 없음 (rc=${2:-?})"
    fi
}

echo ""
echo "--- VANILLA ---"
parse_json "${VANILLA_JSON}" "$VANILLA_RC"
echo "  wall_total_s=${VANILLA_WALL}s  (init+generate)"

echo ""
echo "--- CURRENT (NEO + Phase1A + Phase3) ---"
parse_json "${CURRENT_JSON}" "$CURRENT_RC"
echo "  wall_total_s=${CURRENT_WALL}s  (init+generate)"

echo ""
# ratio 계산
if [ -f "${VANILLA_JSON}" ] && [ -f "${CURRENT_JSON}" ]; then
    python3 -c "
import json
v = json.load(open('${VANILLA_JSON}'))
c = json.load(open('${CURRENT_JSON}'))
v_tps = v.get('output_tps', 0)
c_tps = c.get('output_tps', 0)
v_wall = v.get('generate_wall_s', 0)
c_wall = c.get('generate_wall_s', 0)
if v_tps and c_tps:
    print(f'NEO/vanilla output_tps ratio: {c_tps/v_tps:.3f}x ({c_tps:.1f}/{v_tps:.1f} tps)')
    print(f'NEO/vanilla generate_wall ratio: {c_wall/v_wall:.3f}x ({c_wall:.1f}s/{v_wall:.1f}s)')
else:
    print('비율 계산 불가 (tps=0 또는 파싱 실패)')
" 2>/dev/null
fi

echo ""
echo "[Run-H] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST') DONE → ${OUT_DIR}"
