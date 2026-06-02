#!/usr/bin/env bash
# AGSD 32B sweep — 6 workload × 3 scenario, 500p × 8192in × 8192out, 1 run each.
# 백엔드(8001/8002)+라우터(8000)가 이미 떠 있어야 함.
set -uo pipefail
cd /workspace/host_vllm_hybrid

RUN_DIR="${1:-/workspace/host_vllm_hybrid/vllm_config_perf/gating/runs/agsd_32b_b200_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
PY=/workspace/vllm_dev_prj/bin/python
NP=500; TIN=8192; TOUT=8192; CONC=32
WORKLOADS="sonnet chat code balanced sonnet-heavy code-heavy"
SCENARIOS="vanilla trident AGSD"

echo "[sweep] start $(date) → $RUN_DIR"
echo "[sweep] NP=$NP in=$TIN out=$TOUT conc=$CONC" | tee "$RUN_DIR/_config.txt"

for wl in $WORKLOADS; do
  for sc in $SCENARIOS; do
    out="$RUN_DIR/${sc}__${wl}.json"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    echo "==== $(date '+%H:%M:%S') $sc × $wl ===="
    PYTHONPATH=/workspace/host_vllm_hybrid "$PY" \
      vllm_config_perf/gating/benchmark_workloads.py \
      --scenario "$sc" --workload "$wl" \
      --num-prompts "$NP" --target-input-len "$TIN" --max-tokens "$TOUT" \
      --concurrency "$CONC" --out "$out" \
      >> "$RUN_DIR/sweep.log" 2>&1
    rc=$?
    tps=$("$PY" -c "import json;print(round(json.load(open('$out'))['summary']['output_tps'],1))" 2>/dev/null || echo "ERR rc=$rc")
    echo "     → output_tps=$tps"
  done
done
echo "[sweep] DONE $(date)"
