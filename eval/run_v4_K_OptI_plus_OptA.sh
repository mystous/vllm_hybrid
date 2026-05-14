#!/usr/bin/env bash
# [Plan v4 / Phase K — Option I + Option A] try79 회차.
# Goal: Option I (mirror MIN_BUFFER 보장) 위에서 Option A (cdec_ids =
#       SWAPPED_OUT ∪ mirror, brute force) 활성. mirror size 안정적 →
#       cdec_ids non-empty → sub_batches attach → chain firing 활성.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try79_v4_K_OptI_plus_OptA"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

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

# Option I: MIN_BUFFER guard
export VLLM_NEO_MIRROR_MIN_BUFFER=8

# Option A 활성
export VLLM_NEO_OPTION_A=1
unset VLLM_NEO_OPTION_C

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

echo "[OptI+A] starting Option I + Option A → ${LOG_FILE}"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
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
    > "${LOG_FILE}.stdout" 2>&1
LAUNCHER_RC=$?
echo "[OptI+A] launcher exit=${LAUNCHER_RC}"

echo "[OptI+A] === Phase K-OptI+A gates ==="
echo "[OptI+A] Option I skip first fire:"
grep '\[Option I\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptI+A] Option A / D19 first fire:"
grep '\[Option A / D19\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptI+A] D15+D16 load-aware total fire:"
grep -c '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptI+A] swap_out worker fire:"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptI+A] swap_in worker done (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[OptI+A] D11 OOB precheck:"
grep -c '\[NEO CDEC D11 OOB PRECHECK\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptI+A] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[OptI+A] NEO CDEC CALL counts:"
grep -c '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptI+A] mirror_set_size 마지막 5:"
grep 'mirror_set_size' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo "[OptI+A] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptI+A] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[OptI+A] DONE $(date -Iseconds)"
