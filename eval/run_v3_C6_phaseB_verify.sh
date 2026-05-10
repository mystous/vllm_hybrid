#!/usr/bin/env bash
# [TSK_019 v3 / C6] Phase B 검증 — Phase A + B-1/B-2/B-3 적용 후 표준 workload.
# swap_in 발화 + bidirectional migration + LRU stub 검증.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try51_v3_C6_phaseB_verify_neo_on"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# Phase A+B all defaults — kill switches 모두 unset.
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
export VLLM_NEO_FORCE_PIPELINED=1

echo "[C6] starting Phase B 검증 launcher → ${LOG_FILE}"
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
echo "[C6] launcher exit=${LAUNCHER_RC}"

echo "[C6] result dir: ${OUT_DIR}"
echo "[C6] === Phase B gates ==="
echo "[C6] swap_in dispatch first fire:"
grep 'NEO SWAP_IN dispatch first fire' "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "[C6] swap_in candidates log:"
grep -c 'NEO SWAP_IN.*candidates=' "${LOG_FILE}.stdout" 2>/dev/null
echo "[C6] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error' "${LOG_FILE}.stdout" 2>/dev/null
echo "[C6] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[C6] DONE $(date -Iseconds)"
