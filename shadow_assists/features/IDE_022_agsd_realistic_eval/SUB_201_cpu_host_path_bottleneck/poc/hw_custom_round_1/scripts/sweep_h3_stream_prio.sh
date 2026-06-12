#!/usr/bin/env bash
# H3 — Aux CUDA copy streams at priority=-1 (highest).
# Affects: output_copy_stream (gpu/model_runner), async_output_copy_stream,
# draft_token_ids_copy_stream, valid_sampled_token_count_copy_stream,
# _num_valid_draft_tokens_copy_stream, structured_outputs copy_stream.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib_common.sh"

trap 'echo "[trap]"; exit 130' INT TERM
echo "==== HWC1 H3 stream-prio 1-sweep at $(date -u +%FT%TZ) ===="
wait_gpu_free || true

declare -a EXTRA_ENV=("VLLM_HWC1_STREAM_PRIO=1")
declare -a EXTRA_CLI=()
do_case_nsweep "h3_stream_prio" 1

echo "==== HWC1 H3 complete at $(date -u +%FT%TZ) ===="
wait_gpu_free || true
