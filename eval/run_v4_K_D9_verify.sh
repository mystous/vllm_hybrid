#!/usr/bin/env bash
# [Plan v4 / Phase K — D9 verify] try66 회차.
# Goal: D5 kill switch (VLLM_NEO_DISABLE_D5=1) 활성 → SWAPPED_OUT decode
#       진행 보류 → seq_len 동결 → block_pos OOB 영영 안 됨 → SEGV 0 보장.
#
# Trade-off: 19 항목 중 cdec/fork chain 영역 (#2/#7/#9~#13) ❌. 그러나
#            #1/#5/#14/#18/#19 (KV 메커니즘) ✅ 유지. v4 ground rule
#            "vanilla 회귀 금지" (SEGV 의 극단 회귀) 우선.
#
# Stop condition (PASS):
#   - run 정상 완주 (EngineDead/segfault/AssertionError 0)
#   - swap_out > 0, swap_in done > 0 (KV migration LRU 동작)
#   - result.json 생성
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try66_v4_K_D9_d5_kill_switch"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# [D9] D5 비활성 — SWAPPED_OUT decode 미진행 → SEGV trigger 영영 없음.
export VLLM_NEO_DISABLE_D5=1

# D6/D7 그대로 활성 (swap_in path 작동).
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-D9] starting D9 (D5 kill switch) verify → ${LOG_FILE}"
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
echo "[K-D9] launcher exit=${LAUNCHER_RC}"

echo "[K-D9] === Phase K-D9 gates ==="
echo "[K-D9] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D9] swap_in worker done count (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-D9] CPU-resident skip last:"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D9] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D9] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D9] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[K-D9] DONE $(date -Iseconds)"
