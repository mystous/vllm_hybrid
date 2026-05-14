#!/usr/bin/env bash
# [Plan v4 / Phase K — Option C (D17C) verify] try75 회차.
# Goal: NEO _decide_mode_and_gen_batch Step 3 등가 — adapter.schedule()
#       후반에서 mode_selector.decide_mode 호출하여 load-balanced cdec
#       배포. mirror 의 reqs 를 cpu_decoding_q 로 보고, batches[1].cdec_reqs
#       → cdec_ids.
# Env: VLLM_NEO_OPTION_C=1 활성. Option A 와 둘 중 하나만 활성.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try75_v4_K_D17C_option_c"
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

# Option C 활성 — Option A 는 비활성.
export VLLM_NEO_OPTION_C=1
unset VLLM_NEO_OPTION_A

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

echo "[K-D17C] starting Option C (decide_mode load-balanced) → ${LOG_FILE}"
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
echo "[K-D17C] launcher exit=${LAUNCHER_RC}"

echo "[K-D17C] === Phase K-D17C (Option C) gates ==="
echo "[K-D17C] Option C first fire:"
grep '\[Option C / D17C\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D17C] Option C exception counts:"
grep -c '\[Option C / D17C\] exception' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D17C] D15+D16 load-aware first fire:"
grep '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D17C] D15+D16 load-aware total fire counts:"
grep -c '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D17C] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D17C] swap_in worker done (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-D17C] D11 OOB precheck:"
grep -c '\[NEO CDEC D11 OOB PRECHECK\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D17C] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D17C] NEO CDEC CALL counts:"
grep -c '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D17C] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D17C] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[K-D17C] DONE $(date -Iseconds)"
