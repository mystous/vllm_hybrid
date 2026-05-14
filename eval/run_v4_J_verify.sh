#!/usr/bin/env bash
# [Plan v4 / D3] Phase J 검증 — D0+D1+D2 적용 후 NEO 19 항목 발화 측정.
# Goal: chain firing rate ≥ 50% / [NEO SWAP_OUT CALL] > 0 / cdec 매 step fire.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try60_v4_J_verify_neo_full_migration"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# Plan v4 — 모든 NEO env default ON, kill switch 명시 안 함.
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[D3] starting Plan v4 NEO migration full launcher → ${LOG_FILE}"
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
echo "[D3] launcher exit=${LAUNCHER_RC}"

# === Phase J gates 검증 ===
echo "[D3] result dir: ${OUT_DIR}"
echo "[D3] === Phase J gates ==="
echo "[D3] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[D3] CPU-resident skip log:"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "[D3] swap_out attach (G/H combined):"
grep -c '\[Plan v4 G/H\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[D3] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[D3] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error' "${LOG_FILE}.stdout" 2>/dev/null
echo "[D3] DONE $(date -Iseconds)"
