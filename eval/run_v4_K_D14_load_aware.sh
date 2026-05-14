#!/usr/bin/env bash
# [Plan v4 / Phase K — D14+D15+D16 verify] try73 회차.
# Goal: NEO paper 의 진짜 design 정합 — load-aware sub-batch decision
#       활성화. KV pressure 와 *무관* 한 cdec dispatch trigger.
#
# D14: HeuristicPerfPredictor (interp-free) — perfpredictor.py
# D15: 경량 load-aware decision — adapter.schedule() 안에서 _get_remains
#      식 직접 계산
# D16: cdec_ids 추출 통합 + active swap_out — load_aware 후보를
#      _swap_out_predictive_ids 에 추가
#
# Stop condition (PASS):
#   - run 완주 (EngineDead/segfault/AssertionError 0)
#   - chain firing rate > 5% (try51 의 0.6% 대비 *최소 8×*)
#   - load-aware first fire log 출현
#   - swap_out > try51 영역 (load-aware trigger 발화)
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try73_v4_K_D14_load_aware"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# [D14] heuristic predictor (default 'heuristic') — interp-free.
export VLLM_NEO_PREDICTOR=heuristic

# [D15+D16] load-aware tuning
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32     # 32+ running decode reqs 시 fire
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2  # per-step cap (KV thrashing 회피)

# heuristic constant override (default 적용 — 필요 시 조정)
# export VLLM_NEO_HEURISTIC_LINR_PER_TOKEN_MS=0.05
# export VLLM_NEO_HEURISTIC_PREF_PER_TOKEN_MS=0.10
# export VLLM_NEO_HEURISTIC_GDEC_PER_TOKEN_MS=0.001
# export VLLM_NEO_HEURISTIC_CDEC_PER_TOKEN_PAIR_MS=0.0005

# 기존 v4 D6~D12 stack 도 active (D11 dynamic precheck 가 SEGV 회피)
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

echo "[K-D14] starting D14+D15+D16 (load-aware) verify → ${LOG_FILE}"
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
echo "[K-D14] launcher exit=${LAUNCHER_RC}"

echo "[K-D14] === Phase K-D14 gates ==="
echo "[K-D14] D15+D16 load-aware first fire:"
grep '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D14] D15+D16 load-aware total fire counts:"
grep -c '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D14] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D14] swap_in worker done (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-D14] D11 OOB precheck:"
grep -c '\[NEO CDEC D11 OOB PRECHECK\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D14] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D14] NEO CDEC CALL counts:"
grep -c '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D14] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D14] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[K-D14] DONE $(date -Iseconds)"
