#!/usr/bin/env bash
# IDE_032 결정 실험: hot-80 (CPU 병목) + cold 전량 deferral(N=8) — C32/C64 + GSM40. 비교: hot-80 N=4 (78.9ms TPOT, 332 tok/s), hot-96 N=8 (505)
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
until grep -q "RANDOM_TRACE_DONE\|BOOT_FAIL" $SP/random_trace.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$HOME/projects/vllm_hybrid/eval/results/$(date +%Y%m%d_%H%M%S)_ide032_h80_n8_kill; mkdir -p $B
for N in 8 4; do
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 131072 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 80 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token $N > /tmp/sgl_h80n$N.log 2>&1"
  i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  [ $ok = 1 ] || { echo "H80 N=$N BOOT_FAIL" | tee -a $B/RUN.log; continue; }
  echo "== H80 N=$N boot ${i}s ==" | tee -a $B/RUN.log
  for C in 32 64; do rf=/tmp/h80n${N}_c$C.log; NP=$((C*4))
    docker exec -d vllm-h100 bash -c "timeout 600 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $C --request-rate inf --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
    w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>660)) && break; done; docker cp vllm-h100:$rf $B/h80n${N}_c$C.log 2>/dev/null
    echo "H80 N=$N C$C: $(grep -h 'Output token throughput' $B/h80n${N}_c$C.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/h80n${N}_c$C.log|awk '{print $NF}')" | tee -a $B/RUN.log
  done
  [ $N = 8 ] && python3 $SP/gsm_eval.py $B/h80n8_gsm40.json 40 2>&1 | tail -1 | sed "s/^/H80 N=8 GSM40: /" | tee -a $B/RUN.log
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; echo "H80_N8_KILL_DONE $B"
