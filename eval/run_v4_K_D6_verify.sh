#!/usr/bin/env bash
# [Plan v4 / Phase K — D6 verify] try61 회차.
# Goal: D6 forced swap_in 적용 후 try60-γ 의 pacpu store_kv segfault 회피
#       + SWAP_IN_DONE > 0 활성화 + run 완주 (no EngineDead).
#
# Stop condition (PASS):
#   - run 정상 종료 (EngineDeadError 0, segfault 0, AssertionError 0)
#   - [Plan v4 D4] swap_in attach first fire log 출현 (SWAP_IN > 0)
#   - SWAP_OUT_CALL > 0 유지 (try60 와 동등)
#   - 19 항목 #14 (KV migration LRU) ✅ 진입
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try61_v4_K_D6_verify_forced_swapin"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# [D6] forced swap_in default ON (명시). per-step cap = 2 (KV 한계 영역
# 안전화). swap_in order = oldest (mirror 의 가장 오래된 req 우선 복귀).
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=2
export VLLM_NEO_SWAP_IN_ORDER=oldest

# 다른 NEO env 는 default ON (D0~D5 와 동일 — kill switch unset).
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-D6] starting Plan v4 D6 forced-swap_in verify → ${LOG_FILE}"
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
echo "[K-D6] launcher exit=${LAUNCHER_RC}"

# === Phase K gates 검증 ===
echo "[K-D6] result dir: ${OUT_DIR}"
echo "[K-D6] === Phase K gates ==="
echo "[K-D6] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D6] swap_in attach first fire (D4/D6 path):"
grep '\[Plan v4 D4\] swap_in attach' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D6] swap_in attach total counts:"
grep -c '\[Plan v4 D4\] swap_in attach' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D6] CPU-resident skip log (Plan v4 H):"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "[K-D6] swap_out attach (G/H combined):"
grep -c '\[Plan v4 G/H\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D6] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D6] crash counts (incl. segfault):"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D6] DONE $(date -Iseconds)"
