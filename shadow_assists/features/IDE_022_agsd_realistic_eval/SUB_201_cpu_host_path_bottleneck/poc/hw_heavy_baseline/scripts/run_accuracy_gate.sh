#!/usr/bin/env bash
# Run accuracy gate sequentially against TWO endpoints.
#
# Usage:
#   bash run_accuracy_gate.sh <baseline-tag> <baseline-cli ...> -- <candidate-tag> <candidate-model> <candidate-cli ...>
#
# Example (RedHat FP8 vs vanilla):
#   bash run_accuracy_gate.sh \
#       vanilla meta-llama/Llama-3.1-8B-Instruct -- \
#       redhat_fp8 RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
#
# Captures 50 prompts × 128 tokens via accuracy_capture.py, then diffs.
# Writes:
#   accuracy/<baseline_tag>.jsonl
#   accuracy/<candidate_tag>.jsonl
#   accuracy/gate_<candidate_tag>_vs_<baseline_tag>.json
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
OUTDIR=$ROOT/accuracy
mkdir -p "$OUTDIR"

# parse: baseline_tag baseline_model [--cli ...] -- candidate_tag candidate_model [--cli ...]
BASE_TAG=$1; shift
BASE_MODEL=$1; shift
BASE_CLI=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do BASE_CLI+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
CAND_TAG=$1; shift
CAND_MODEL=$1; shift
CAND_CLI=()
while [ $# -gt 0 ]; do CAND_CLI+=("$1"); shift; done

export ROOT MODEL=  # placeholder; we set per-case below
export PORT=8099   # dedicated gate port to avoid clashes
export TP=8
export MAX_MODEL_LEN=8192
export CONC=1
export NPROMPT=50
export MAX_TOKENS=128
export N_SWEEPS=1
export EXTRA_LOG_TAG=hwhgate
source "$SCRIPT_DIR/lib_heavy.sh"

VPY=/workspace/vllm_dev_prj/bin/python
GATE_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_heavy_baseline/scripts
PARQUET=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_1/sharegpt500.parquet

capture_one() {
    local tag=$1 model=$2; shift 2
    local cli=("$@")
    echo "=== capture $tag model=$model cli=${cli[*]:-} ==="
    MODEL=$model
    EXTRA_ENV=()
    EXTRA_CLI=("${cli[@]}")
    start_one "gate_${tag}" || { echo "[gate] $tag boot fail"; return 1; }
    "$VPY" "$GATE_DIR/accuracy_capture.py" \
        --url "http://127.0.0.1:$PORT" \
        --model "$model" \
        --parquet "$PARQUET" \
        --n-prompts 50 --seed 43 --max-tokens 128 \
        --out "$OUTDIR/${tag}.jsonl"
    stop_one "gate_${tag}"
    return 0
}

capture_one "$BASE_TAG" "$BASE_MODEL" "${BASE_CLI[@]}" || exit 1
sleep 5; wait_gpu_free || true
capture_one "$CAND_TAG" "$CAND_MODEL" "${CAND_CLI[@]}" || exit 1
sleep 3

"$VPY" "$GATE_DIR/accuracy_diff.py" \
    --baseline "$OUTDIR/${BASE_TAG}.jsonl" \
    --candidate "$OUTDIR/${CAND_TAG}.jsonl" \
    --out "$OUTDIR/gate_${CAND_TAG}_vs_${BASE_TAG}.json"
