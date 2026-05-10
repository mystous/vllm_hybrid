#!/usr/bin/env bash
# [Plan v4 / Phase K — D8 verify] try64 회차.
# Goal: D6+D7+D8 누적 적용. D8 = block_count 가드 (D5 fix 의 OOB 안전선).
#       try60-γ/62/63 SEGV root (mirror 잔류 reqs 의 seq_len 누적 →
#       block_pos OOB) 회피.
#
# Stop condition (PASS):
#   - run 정상 종료 (EngineDeadError/segfault 0)
#   - swap_in worker done > 100 (반복 fire)
#   - chain firing 발화 (cdec dispatch 의 mirror 안전 영역 reqs 만 처리)
#   - 19 항목 #14, #18, #19 ✅
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try64_v4_K_D8_block_count_guard"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# D8 의 핵심은 코드 가드. env 는 cap 4 / mirror 64 (보수). cycle 16 step
# 이지만 D8 가 OOB 영역 진입 자체를 차단해서 안전.
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-D8] starting D6+D7+D8 verify → ${LOG_FILE}"
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
echo "[K-D8] launcher exit=${LAUNCHER_RC}"

echo "[K-D8] result dir: ${OUT_DIR}"
echo "[K-D8] === Phase K-D8 gates ==="
echo "[K-D8] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D8] safe_max stash first log (D8):"
grep '\[Plan v4 G+D8\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D8] swap_in worker done count (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-D8] D7 race guard fire counts:"
grep -c '\[Plan v4 D7\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D8] CPU-resident skip last:"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D8] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D8] crash counts (incl. segfault):"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D8] result.json check:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성 — 회차 미완주)"
echo "[K-D8] DONE $(date -Iseconds)"
