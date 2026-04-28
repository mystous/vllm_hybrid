#!/usr/bin/env bash
# =============================================================================
# run_prod_partial_profile_64k.sh — partition path 의 layer breakdown 측정.
#
# VLLM_PARTIAL_ATTN_PROFILE=1 으로 flash_attn.py 의 hot_cold_attention 안의
# layer 단위 d2h_ms / kernel_ms / h2d_ms / merge_ms / total_ms breakdown 을
# stderr 에 출력. 어느 step 이 dominant 인지 정량 — *진짜 코드 결함* 위치 식별.
#
# 짧게 — 5 prompts × 65536 input × 4 output. 가치 영역 진입 보장 + 시간 단축.
# split_on env (cold-tier ON, IDE_006 ON) 만 측정 (baseline 비교 불필요).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_hwtag.sh"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib64/libcuda.so.1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# 핵심 — partition path 의 layer breakdown 출력.
export VLLM_PARTIAL_ATTN_PROFILE=1
# fallback 은 끔 (deadline 매우 큼) — partition path 만 보기 위함.
export VLLM_COLD_KV_FALLBACK_DEADLINE_MS=0

PYTHON="${PYTHON:-/workspace/vllm_dev_prj/bin/python}"
TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${SCRIPT_DIR}/results/${TS}_${HW_TAG}_partial_profile_64k"
mkdir -p "${OUT_DIR}/e2e_artifacts"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

{
    echo "# ${TS}_${HW_TAG}_partial_profile_64k"
    echo "- VLLM_PARTIAL_ATTN_PROFILE=1"
    echo "- VLLM_COLD_KV_FALLBACK_DEADLINE_MS=0 (fallback 비활성, partition only)"
    echo "- max-prompts=5, input=65536, output=4"
} > "${OUT_DIR}/README.md"

E2E_RC=0
HW_TAG="${HW_TAG}" stdbuf -oL -eL "${PYTHON}" -u "${SCRIPT_DIR}/run_e2e_accuracy.py" \
    --baseline-env "${SCRIPT_DIR}/envs/ide006_cold_tier_only_value_64k.env" \
    --split-on-env "${SCRIPT_DIR}/envs/ide006_cold_kv_split_on_value_64k.env" \
    --max-prompts 5 \
    --max-tokens 4 \
    --logprobs 0 \
    --split-on-only \
    --output-dir "${OUT_DIR}/e2e_artifacts" \
    > >(tee "${OUT_DIR}/e2e.log") 2> >(tee "${OUT_DIR}/e2e.stderr.log" >&2) || E2E_RC=$?
log "  e2e exit=${E2E_RC}"

# Profile lines 추출 + 통계.
echo
echo "===profile breakdown summary==="
"${PYTHON}" <<PYEOF
import re, statistics
import sys

LINES = []
with open("${OUT_DIR}/e2e.stderr.log") as f:
    for line in f:
        if "[IDE_006/TSK_004 profile" in line:
            m = re.search(r"d2h_ms=([\d.]+) kernel_ms=([\d.]+) h2d_ms=([\d.]+) merge_ms=([\d.]+) total_ms=([\d.]+)", line)
            if m:
                LINES.append([float(x) for x in m.groups()])

if not LINES:
    print("NO profile lines — VLLM_PARTIAL_ATTN_PROFILE 가 효과 없음 또는 cold path 발화 안 함")
    sys.exit(0)

n = len(LINES)
labels = ["d2h_ms", "kernel_ms", "h2d_ms", "merge_ms", "total_ms"]
print(f"profile samples: {n}")
print(f"{'metric':<10} {'mean':>8} {'median':>8} {'p90':>8} {'max':>8} {'sum':>10}")
for i, lbl in enumerate(labels):
    vals = sorted([row[i] for row in LINES])
    print(f"{lbl:<10} {statistics.mean(vals):>8.2f} {statistics.median(vals):>8.2f} "
          f"{vals[int(0.9*len(vals))]:>8.2f} {max(vals):>8.2f} {sum(vals):>10.1f}")
print()
print("breakdown of mean total_ms:")
mt = statistics.mean([r[4] for r in LINES])
for i, lbl in enumerate(["d2h", "kernel", "h2d", "merge"]):
    m = statistics.mean([r[i] for r in LINES])
    pct = 100 * m / mt if mt > 0 else 0
    print(f"  {lbl:<8} {m:>6.2f} ms ({pct:>5.1f}%)")
PYEOF

exit 0
