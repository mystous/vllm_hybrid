#!/usr/bin/env bash
# H8 (KV fp8) accuracy gate vs baseline.
# 1) Launch baseline (no fp8) → capture logprobs on 50 prompts
# 2) Stop, launch candidate (fp8) → capture logprobs on same 50 prompts
# 3) Compare per-token logprob max-abs-diff and per-sequence PPL relative diff
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM

declare -a EXTRA_ENV=()
declare -a EXTRA_CLI=()

NPROMPT_ACC=50

# Step 1: baseline (no fp8)
echo "==== Step 1: baseline (no fp8) ====" | tee -a $LOGS/accuracy_gate.log
EXTRA_ENV=()
EXTRA_CLI=()
if start_one "acc_baseline"; then
    "$VPY" "$SCRIPT_DIR/capture_logprobs.py" \
        --port $PORT --model "$MODEL" --parquet "$SAMPLED" \
        --n $NPROMPT_ACC --max-tokens 64 \
        --out $RUNS/acc_baseline_logp.jsonl \
        | tee -a $LOGS/accuracy_gate.log
    stop_one "acc_baseline"
else
    echo "Baseline boot failed" | tee -a $LOGS/accuracy_gate.log
    stop_one "acc_baseline"
    exit 1
fi

# Step 2: candidate (fp8)
echo "==== Step 2: candidate (kv fp8) ====" | tee -a $LOGS/accuracy_gate.log
EXTRA_ENV=()
EXTRA_CLI=(--kv-cache-dtype fp8)
if start_one "acc_kv_fp8"; then
    "$VPY" "$SCRIPT_DIR/capture_logprobs.py" \
        --port $PORT --model "$MODEL" --parquet "$SAMPLED" \
        --n $NPROMPT_ACC --max-tokens 64 \
        --out $RUNS/acc_kv_fp8_logp.jsonl \
        | tee -a $LOGS/accuracy_gate.log
    stop_one "acc_kv_fp8"
else
    echo "Candidate boot failed" | tee -a $LOGS/accuracy_gate.log
    stop_one "acc_kv_fp8"
    exit 1
fi

# Step 3: compare offline
"$VPY" "$SCRIPT_DIR/compare_logprobs.py" \
    --a $RUNS/acc_baseline_logp.jsonl \
    --b $RUNS/acc_kv_fp8_logp.jsonl \
    --out $RUNS/acc_kv_fp8_gate.json \
    | tee -a $LOGS/accuracy_gate.log
