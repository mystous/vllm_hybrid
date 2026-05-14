#!/usr/bin/env bash
# [TSK_019 v3 / C8] Phase D 최종 검증 — vanilla baseline 측정 (NEO OFF) 후
# C6 NEO ON 결과와 비교. throughput win + token correctness gate.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try52_v3_C8_phaseD_vanilla_baseline"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# 모든 NEO env 비활성 — 순수 vanilla.
unset VLLM_NEO_FORCE_PIPELINED VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_FUSED_RMSNORM
unset VLLM_NEO_DISABLE_SWAP_IN VLLM_NEO_LRU_FALLBACK_FIFO

echo "[C8] starting vanilla baseline (NEO OFF) → ${LOG_FILE}"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --num-prompts 500 \
    --target-input-len 8192 \
    --max-tokens 8192 \
    --async-scheduling \
    --enforce-eager false \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" \
    --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1
LAUNCHER_RC=$?
echo "[C8] vanilla launcher exit=${LAUNCHER_RC}"

echo "[C8] vanilla result: ${OUT_DIR}"
cat "${OUT_DIR}/result.json" 2>/dev/null

echo ""
echo "[C8] === Phase D summary ==="
echo "[C8] Comparing to NEO ON (Phase B C6) result..."
NEO_DIR=$(ls -td "${ROOT_DIR}/eval/results/"*try51_v3_C6* 2>/dev/null | head -1)
if [ -n "${NEO_DIR}" ]; then
    echo "[C8] NEO ON dir: ${NEO_DIR}"
    cat "${NEO_DIR}/result.json" 2>/dev/null
fi
echo "[C8] DONE $(date -Iseconds)"
