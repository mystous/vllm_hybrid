#!/usr/bin/env bash
# [TSK_019 v3 / C3] Phase A 검증 — 500p × 50:50 표준 workload.
# Phase A-0 (CPU pool 2x + CDEC default option ready) + A-1 (RMSNorm fused)
# + A-2 (try22 skip default off) 적용 후 NEO ON 측정.
#
# Phase A 종료 조건 모든 gate 검증:
# - chain firing (NEO FORK STAT active 비율) ≥ 80%
# - OOM 0 / AssertionError 0 / CUDA assert 0
# - run 정상 종료
# - output_tps > vanilla
# - swap_out → CDEC 비율 ≥ 90% (vanilla preempt 폴백 ≤ 10%)
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try50_v3_C3_phaseA_verify_neo_on"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# Phase A 변경 활성 — kill switch 모두 unset (default 활성).
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM
# CDEC default 활성 위해 force-pipelined 옵트인 (또는 env).
export VLLM_NEO_FORCE_PIPELINED=1

echo "[C3] starting NEO ON launcher → ${LOG_FILE}"
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
echo "[C3] launcher exit=${LAUNCHER_RC}"

# Summary — Phase A gate 검증.
echo "[C3] result dir: ${OUT_DIR}"
echo "[C3] === Phase A gates ==="
echo "[C3] CPU pool sizing log:"
grep 'NeoCpuKvBuffer sizing' "${LOG_FILE}.stdout" 2>/dev/null | head -2
echo "[C3] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error' \
    "${LOG_FILE}.stdout" 2>/dev/null
echo "[C3] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[C3] swap activity:"
grep -cE 'swap_out|swap_in|cdec' "${LOG_FILE}.stdout" 2>/dev/null
echo "[C3] DONE $(date -Iseconds)"
