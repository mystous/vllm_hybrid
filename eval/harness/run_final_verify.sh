#!/usr/bin/env bash
# 최종 구성 검증: 부팅 → GSM 100 → C=32 정상 상태(45초 후) 프로파일 → C16/C32/C64 벤치
# env: HOT(96) MF(0.92) MTT(24576) EXTRA_ARGS TAG
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
REPO=$HOME/projects/vllm_hybrid
HOT="${HOT:-96}"; MF="${MF:-0.92}"; MTT="${MTT:-24576}"; TAG="${TAG:-final}"
TS=$(date +%Y%m%d_%H%M%S); BASE=$REPO/eval/results/${TS}_ide030_final_${TAG}; mkdir -p "$BASE"
LOG=/tmp/sgl_final_${TAG}.log

docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static $MF --max-total-tokens $MTT --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $HOT --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 ${EXTRA_ARGS:-}"
echo "CMD: $CMD" | tee "$BASE/RUN.log"
echo "== BOOT $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"
docker exec -d sgl-kt bash -c "$CMD > $LOG 2>&1"
i=0; while ((i<1800)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "HEALTH_OK at ${i}s" | tee -a "$BASE/RUN.log"; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || { echo BOOT_FAIL | tee -a "$BASE/RUN.log"; exit 1; }; sleep 10; i=$((i+10)); done

echo "== GREEDY 4 ==" | tee -a "$BASE/RUN.log"
for P in "The capital of France is" "def fibonacci(n):\n    " "The three primary colors are" "Q: What is 17*23? A:"; do
  curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"q480\",\"prompt\":\"$P\",\"max_tokens\":24,\"temperature\":0}" \
   | python3 -c "import json,sys; r=json.load(sys.stdin)['choices'][0]; print(repr(r['text'][:80]), r.get('finish_reason'))" | tee -a "$BASE/RUN.log"
done
echo "== GSM 100 $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"
python3 $SP/gsm_eval.py "$BASE/gsm100.json" 100 2>&1 | tail -2 | tee -a "$BASE/RUN.log"

bench() { docker exec vllm-h100 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions \
    --model q480 --tokenizer "$TOK" --dataset-name sonnet --dataset-path /tmp/sonnet.txt \
    --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 \
    --num-prompts $2 --max-concurrency $1 --request-rate inf --seed 42 \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > "$BASE/bench_c$1.log" 2>&1
  echo "C=$1: $(grep -E 'Output token throughput' "$BASE/bench_c$1.log" | awk '{print $NF}') tok/s, TPOT $(grep 'Mean TPOT' "$BASE/bench_c$1.log" | awk '{print $NF}') ms" | tee -a "$BASE/RUN.log"; }

echo "== C=32 + 정상상태 프로파일 $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"
docker exec sgl-kt bash -c 'mkdir -p /tmp/sgl_prof2 && rm -rf /tmp/sgl_prof2/*'
bench 32 160 &
BPID=$!; sleep 45
curl -s -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' -d '{"output_dir":"/tmp/sgl_prof2","num_steps":6,"activities":["CPU","GPU"]}' >/dev/null; sleep 15
curl -s -X POST http://127.0.0.1:30000/stop_profile >/dev/null; wait $BPID
docker cp sgl-kt:/tmp/sgl_prof2/. "$BASE/" 2>/dev/null; ls "$BASE" | grep -c trace.json.gz | tee -a "$BASE/RUN.log"
echo "== C=16 ==" | tee -a "$BASE/RUN.log"; bench 16 96
echo "== C=64 ==" | tee -a "$BASE/RUN.log"; bench 64 192
echo "== DONE $(date +%H:%M:%S) ==" | tee -a "$BASE/RUN.log"; echo "$BASE"; echo FINAL_VERIFY_DONE
