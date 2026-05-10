#!/usr/bin/env bash
# [Plan v4 / Phase K — D8v2 verify] try65 회차.
# Goal: D8 의 fallback hole 제거 — stash 없는 reqs 는 D5 fix 미발동.
#       try64 의 SEGV 1회 잔존 (stash 안 된 swap-out path) root 회피.
#
# Stop condition (PASS):
#   - run 완주 (EngineDeadError/segfault 0)
#   - swap_in done > 100
#   - 19 항목 #14, #18, #19 ✅
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try65_v4_K_D8v2_safe_fallback"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-D8v2] starting D8 v2 (safe fallback) verify → ${LOG_FILE}"
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
echo "[K-D8v2] launcher exit=${LAUNCHER_RC}"

echo "[K-D8v2] === Phase K-D8v2 gates ==="
echo "[K-D8v2] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D8v2] D8 stash first log:"
grep '\[Plan v4 G+D8\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D8v2] swap_in worker done count (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-D8v2] D7 race guard fire counts:"
grep -c '\[Plan v4 D7\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D8v2] CPU-resident skip last:"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D8v2] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D8v2] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D8v2] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[K-D8v2] DONE $(date -Iseconds)"
