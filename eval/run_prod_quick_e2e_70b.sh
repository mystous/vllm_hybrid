#!/usr/bin/env bash
# =============================================================================
# run_prod_quick_e2e_70b.sh — Llama-3.3-70B + TP=8 e2e 짧은 cold-path 발화
#   검증 (offline 환경 정합).
#
# 본 wrapper 는 기존 run_prod_cold_verify.sh 를 그대로 호출하되:
#   - LD_PRELOAD=/usr/lib64/libcuda.so.1  (prod libcuda 경로 workaround)
#   - HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (방화벽 환경, 외부 HF Hub 차단)
#   - NUM_PROMPTS / OUTPUT_LEN 짧은 default (cold path firing 검증용)
#
# 모델은 *long_ctx env 의 meta-llama/Llama-3.3-70B-Instruct 사용 (사전
# /root/.cache/huggingface 에 배포됨, 263 GB).
#
# 실행 시간 견적: ~5~10 분
#   (TP=8 70B startup 3~5 분 + prefill 1~3 분 + 짧은 decode + monitor)
#
# Usage:
#   bash eval/run_prod_quick_e2e_70b.sh
#   NUM_PROMPTS=20 OUTPUT_LEN=8 bash eval/run_prod_quick_e2e_70b.sh
#
# 산출물 — run_prod_cold_verify.sh 의 표준 layout:
#   eval/results/<TS>_<HW_TAG>_cold_verify/
#     ├── README.md / isa_info.txt
#     ├── e2e.log / e2e.stderr.log
#     ├── e2e_artifacts/split_on.json
#     └── monitor_cpu.csv / monitor_gpu.csv / monitor.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# libcuda.so.1 search path workaround on this prod box.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"

# 외부망 차단 (방화벽 환경). HF Hub 호출 자체를 막아 gated/offline race
# 회귀를 사전 봉쇄. 로컬 /root/.cache/huggingface 만 사용.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# 짧게. cold path 발화는 KV pool overflow 가 필요하므로 NUM_PROMPTS 는
# 충분한 수준 유지 (기존 cold_verify 50/50 success 회차 기준 = 50).
export NUM_PROMPTS="${NUM_PROMPTS:-50}"
export OUTPUT_LEN="${OUTPUT_LEN:-16}"

# TSK_011 §4.1/4.2 — deadline-aware cold path + GPU full FA fallback.
# layer 당 hot FA 시간 (~25 ms @ 70B+TP=8) 의 4× 정도로 default 100 ms.
# 너무 짧으면 fallback 빈번 → throughput 저하, 너무 길면 fallback 안 발동.
# 0 이면 fallback 비활성 (pre-TSK_011 동작 — 무한 sync 대기).
export VLLM_COLD_KV_FALLBACK_DEADLINE_MS="${VLLM_COLD_KV_FALLBACK_DEADLINE_MS:-100}"

echo "[$(date '+%H:%M:%S')] dispatching to run_prod_cold_verify.sh"
echo "[$(date '+%H:%M:%S')]   MODEL: meta-llama/Llama-3.3-70B-Instruct (TP=8)"
echo "[$(date '+%H:%M:%S')]   NUM_PROMPTS=${NUM_PROMPTS} OUTPUT_LEN=${OUTPUT_LEN}"
echo "[$(date '+%H:%M:%S')]   offline mode: HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"

exec bash "${SCRIPT_DIR}/run_prod_cold_verify.sh"
