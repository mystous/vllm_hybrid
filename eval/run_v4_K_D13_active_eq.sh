#!/usr/bin/env bash
# [Plan v4 / Phase K — D13 active equilibrium] try72 회차.
# Goal: NEO bistability 의 결정적 trigger 식별. KV threshold 를 1.0 → 0.90
#       으로 낮춰 swap_out cascade 를 *조기* trigger → active 평형으로
#       자동 수렴 reproducible 검증.
#
# try68 만 우연히 active 평형 진입 (chain firing 6.4%), try69~71 은
# inactive (chain 0). D13 = swap_out 첫 trigger 시점을 결정적으로
# 조기화 → cascade 형성 → active 평형 lock-in.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try72_v4_K_D13_active_equilibrium"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

unset VLLM_NEO_DISABLE_D5
unset VLLM_NEO_D12_TOKEN_MARGIN

# [D13] swap_out trigger threshold 1.0 → 0.90 (KV 90% 도달 시 swap_out 시작).
# default 1.0 = KV pool full 상태에서만 swap_out → bistability 의 inactive
# 평형 위험. 0.90 = 더 일찍 fire → cascade 형성 → active 평형 자동 진입.
export VLLM_NEO_PREDICTIVE_THRESHOLD=0.90

export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-D13] starting D13 active-equilibrium (KV threshold 0.90) → ${LOG_FILE}"
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
echo "[K-D13] launcher exit=${LAUNCHER_RC}"

echo "[K-D13] === Phase K-D13 gates ==="
echo "[K-D13] D11 OOB precheck total fires:"
grep -c '\[NEO CDEC D11 OOB PRECHECK\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D13] D8 stash first log:"
grep '\[Plan v4 G+D8\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D13] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D13] swap_in worker done (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-D13] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D13] NEO CDEC CALL counts:"
grep -c '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D13] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D13] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[K-D13] DONE $(date -Iseconds)"
