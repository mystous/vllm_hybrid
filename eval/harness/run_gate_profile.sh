#!/usr/bin/env bash
# 현재 떠 있는 서버(:30000)에 대해 (1) GSM8K 40문항 게이트 (2) C=32 부하 중 torch profiler 트레이스 수집
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
TAG="${TAG:-cur}"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide030_gate_${TAG}
mkdir -p "$BASE"
curl -sf http://127.0.0.1:30000/health >/dev/null || { echo "NO SERVER"; exit 1; }

echo "== GSM 40 $(date +%H:%M:%S) ==" | tee "$BASE/RUN.log"
python3 $SP/gsm_eval.py "$BASE/gsm40.json" 40 2>&1 | tail -3 | tee -a "$BASE/RUN.log"

if [ "${PROFILE:-1}" = "1" ]; then
  echo "== PROFILE (C=32 부하 중 decode 8 스텝) $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"
  docker exec sgl-kt mkdir -p /tmp/sgl_prof && docker exec sgl-kt rm -rf /tmp/sgl_prof/*
  # 부하 먼저 띄우고 3초 뒤 프로파일 시작 (decode 정상 구간)
  docker exec vllm-h100 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions \
    --model q480 --tokenizer "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(ls $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/ | head -1)" \
    --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 \
    --num-prompts 96 --max-concurrency 32 --request-rate inf --seed 42 > "$BASE/bench_during_profile.log" 2>&1 &
  BPID=$!
  sleep 12
  curl -s -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
    -d '{"output_dir":"/tmp/sgl_prof","num_steps":8,"activities":["CPU","GPU"],"with_stack":false,"record_shapes":false}' | tee -a "$BASE/RUN.log"; echo
  sleep 20
  curl -s -X POST http://127.0.0.1:30000/stop_profile | tee -a "$BASE/RUN.log"; echo
  wait $BPID
  docker exec sgl-kt bash -c 'ls -la /tmp/sgl_prof/ | head; du -sh /tmp/sgl_prof' | tee -a "$BASE/RUN.log"
fi
echo "== DONE $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"; echo "$BASE"
