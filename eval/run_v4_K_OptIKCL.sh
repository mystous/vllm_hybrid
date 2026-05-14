#!/usr/bin/env bash
# [Plan v4 / Phase K — Option I + K + C + L] try81 회차.
# Goal: I (MIN_BUFFER) + K (D10 완화) + C (decide_mode) + L (매 step 증분
#       CPU block alloc — NEO 정통 정합). 진짜 chain firing 활성화 + swap-in
#       shape mismatch 회피 + D11 OOB 회피.
# NEO source 정합: pacpu/core.h:21 block_pos=(seq_len-1)/BS,
#                   pacpu/core.h:22 block_table[block_pos] valid 보장.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try81_v4_K_OptIKCL"
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

# I + K + C + L
export VLLM_NEO_MIRROR_MIN_BUFFER=8
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
unset VLLM_NEO_OPTION_A

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

echo "[OptIKCL] starting I+K+C+L → ${LOG_FILE}"
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
echo "[OptIKCL] launcher exit=${LAUNCHER_RC}"

echo "[OptIKCL] === Phase K-OptIKCL gates ==="
echo "[OptIKCL] Option L EXTEND first fire:"
grep '\[NEO BUF EXTEND\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptIKCL] Option L EXTEND total counts:"
grep -c '\[NEO BUF EXTEND\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKCL] Option L EXTEND FAIL counts:"
grep -c '\[NEO BUF EXTEND FAIL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKCL] Option C / D17C first fire:"
grep '\[Option C / D17C\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptIKCL] D11 OOB precheck (Option L 성공 시 0 근처 기대):"
grep -c '\[NEO CDEC D11 OOB PRECHECK\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKCL] swap-in shape mismatch fail (Option L 성공 시 0 기대):"
grep -c 'swap-in.*shape mismatch' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKCL] D15+D16 first fire:"
grep '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptIKCL] swap_out worker fire:"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKCL] NEO CDEC CALL counts (worker side):"
grep -c '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKCL] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[OptIKCL] FORK active>0 count:"
grep -E 'NEO FORK STAT.*active=[1-9]' "${LOG_FILE}.stdout" 2>/dev/null | wc -l
echo "[OptIKCL] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKCL] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[OptIKCL] DONE $(date -Iseconds)"
