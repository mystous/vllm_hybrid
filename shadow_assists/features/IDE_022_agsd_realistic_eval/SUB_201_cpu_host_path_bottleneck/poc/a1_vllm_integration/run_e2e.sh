#!/usr/bin/env bash
# A1 e2e sweep — 한 mode 부팅, correctness collect + e2e tps 측정, kill.
# Usage:  run_e2e.sh MODE MODEL [TP]
#   MODE  = vanilla | cpu_amx_draft | suffix
#   MODEL = HF model id  (e.g. meta-llama/Llama-3.1-8B-Instruct)
#   TP    = tensor-parallel size (default 8)
set -uo pipefail
MODE="${1:?usage: run_e2e.sh MODE MODEL [TP]}"
MODEL="${2:?usage: run_e2e.sh MODE MODEL [TP]}"
TP="${3:-8}"

ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/a1_vllm_integration
PORT="${PORT:-8005}"
VPY=/workspace/vllm_dev_prj/bin/python
RUNNER=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py
SAMPLED=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full/sharegpt200.parquet
RUNS=$ROOT/runs
mkdir -p "$RUNS"

TAG="${MODE}_$(basename "$MODEL" | tr ' /' '__')"
log(){ echo "[$(date '+%H:%M:%S')] [$TAG] $*"; }

# 1) correctness collect.
# SKIP_COLLECT=1 → skip; CORR_N (default 100), CORR_MAXTOK (default 64),
# CORR_CONC (default 16) — cpu_amx_draft 처럼 CPU forward 가 매우 느린
# (3 tok/s/req) 모드는 작은 sample 권장.
if [ "${SKIP_COLLECT:-0}" != "1" ]; then
  log "=== correctness gate collect ($MODE) ==="
  $VPY $ROOT/correctness_gate.py collect --port "$PORT" --model "$MODEL" \
    --out "$RUNS/correctness_${TAG}.jsonl" \
    --n-prompts "${CORR_N:-100}" \
    --max-tokens "${CORR_MAXTOK:-64}" \
    --conc "${CORR_CONC:-16}" \
    > "$RUNS/correctness_${TAG}.collect.log" 2>&1
  tail -5 "$RUNS/correctness_${TAG}.collect.log"
fi

# 2) e2e tps — sharegpt200 × conc=32 × max-tokens=8192 (env 로 조정 가능)
TPUT_LIMIT="${TPUT_LIMIT:-200}"
TPUT_MAX="${TPUT_MAX:-8192}"
TPUT_CONC="${TPUT_CONC:-32}"
log "=== e2e tps sweep (${TPUT_LIMIT}p × conc=${TPUT_CONC} × max-tokens=${TPUT_MAX}) ==="
METHOD_TAG="$MODE"
TPUT_OUT="$RUNS/tput_${TAG}.json"
TPUT_RAW="$RUNS/tput_${TAG}.raw.jsonl"
$VPY "$RUNNER" \
  --in "$SAMPLED" --method "$METHOD_TAG" \
  --model "$MODEL" --port "$PORT" \
  --max-tokens "$TPUT_MAX" --concurrency "$TPUT_CONC" \
  --limit "$TPUT_LIMIT" --shuffle \
  --out "$TPUT_OUT" --raw "$TPUT_RAW" \
  > "$RUNS/tput_${TAG}.log" 2>&1
tail -20 "$RUNS/tput_${TAG}.log"

log "=== summary ==="
cat "$TPUT_OUT" | $VPY -m json.tool | head -40
