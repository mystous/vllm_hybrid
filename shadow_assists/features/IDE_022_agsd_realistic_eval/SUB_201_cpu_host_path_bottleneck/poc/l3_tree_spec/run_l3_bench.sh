#!/usr/bin/env bash
# L3 — run sharegpt-100p × conc=8 × max-tok 256 against the L3 launcher.
#
# Usage:  ./run_l3_bench.sh baseline   ./run_l3_bench.sh tree
#
# Output: out/<mode>.json (summary), out/<mode>.raw.jsonl (per-request)
#         + scrapes /metrics for spec accept rate before/after.

set -euo pipefail

MODE="${1:-baseline}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="/workspace/host_vllm_hybrid"
PYBIN="${L3_PYBIN:-/workspace/vllm_dev_prj/bin/python}"
PARQUET="${L3_PARQUET:-$ROOT/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet}"
MODEL="${L3_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
PORT="${L3_PORT:-8003}"
CONC="${L3_CONC:-8}"
MAX_TOK="${L3_MAX_TOK:-256}"
N="${L3_NUM_PROMPTS:-100}"
CORPUS="${L3_CORPUS:-sharegpt}"
TAG="${L3_TAG:-$MODE}"

OUT_DIR="$HERE/out"
mkdir -p "$OUT_DIR"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"

"$PYBIN" -m vllm_config_perf.gating.realistic_eval.throughput_runner \
  --in "$PARQUET" \
  --method "l3_$MODE" \
  --model "$MODEL" \
  --port "$PORT" \
  --max-tokens "$MAX_TOK" \
  --concurrency "$CONC" \
  --limit "$N" \
  --corpus "$CORPUS" \
  --no-stream \
  --out "$OUT_DIR/${TAG}.json" \
  --raw "$OUT_DIR/${TAG}.raw.jsonl"

# Also pull the patched-side L3 counters from the engine's prometheus
# scrape — vllm exposes only spec α; the in-proposer counters (tree_accept
# upper bound) live in-process and we surface them via the stats dump file.
if [ -n "${VLLM_L3_TREE_STATS_PATH:-}" ]; then
  echo "[l3] stats file: $VLLM_L3_TREE_STATS_PATH ($(wc -l < "$VLLM_L3_TREE_STATS_PATH") records)"
fi
