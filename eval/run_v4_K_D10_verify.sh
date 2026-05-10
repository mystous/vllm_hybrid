#!/usr/bin/env bash
# [Plan v4 / Phase K — D10 verify] try67 회차.
# Goal: 모든 SWAPPED_OUT reqs 의 num_new_tokens 를 safe_max 너머로 가지
#       못하게 clamp. D5 분기 외부 가드 — try60-γ ~ try66 의 잔존 SEGV
#       root (num_new_tokens > 0 인 SWAPPED_OUT reqs) 회피.
#
# Stop condition (PASS):
#   - run 정상 완주 (EngineDead/segfault/AssertionError 0)
#   - 19 항목: chain firing + cdec dispatch 활성 (D5 부여 분기 살아있음)
#   - swap_in done > 0 sustained
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try67_v4_K_D10_global_swapped_clamp"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# D9 (D5 kill switch) 비활성 — D5 fix 살리고 D10 가 가드 담당.
unset VLLM_NEO_DISABLE_D5

export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-D10] starting D10 (global SWAPPED_OUT clamp) verify → ${LOG_FILE}"
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
echo "[K-D10] launcher exit=${LAUNCHER_RC}"

echo "[K-D10] === Phase K-D10 gates ==="
echo "[K-D10] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D10] D8 stash first log:"
grep '\[Plan v4 G+D8\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D10] swap_in worker done count (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-D10] D7 race guard fire counts:"
grep -c '\[Plan v4 D7\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D10] CPU-resident skip last:"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D10] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D10] NEO CDEC CALL counts:"
grep -c '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D10] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D10] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[K-D10] DONE $(date -Iseconds)"
