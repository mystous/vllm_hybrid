#!/usr/bin/env bash
# [Plan v4 / D3 small] 19 항목 발화 verification — 작은 workload (CPU pacpu
# 가 GPU 보다 200x 느려서 표준 500p 는 44h+ 소요. 50p × 2048 max_tok 로 축소).
# Goal: 19 항목 발화 demo (성능 측정 X — 발화 verify only).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try60_v4_J_verify_neo_small"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN

echo "[D3-small] starting → ${LOG_FILE}"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --num-prompts 50 \
    --target-input-len 1024 \
    --max-tokens 1024 \
    --enable-neo-asymmetric \
    --async-scheduling \
    --enforce-eager false \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 4096 \
    --log-file "${LOG_FILE}" \
    --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1
LAUNCHER_RC=$?
echo "[D3-small] launcher exit=${LAUNCHER_RC}"

echo "[D3-small] === gates ==="
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
grep '\[Plan v4 G/H\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
grep 'NEO SWAP\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -2
grep -c 'NEO CDEC CALL' "${LOG_FILE}.stdout" 2>/dev/null
echo "[D3-small] DONE $(date -Iseconds)"
