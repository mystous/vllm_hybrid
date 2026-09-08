#!/usr/bin/env bash
# 최종 구성 (hot-96 N=8, CF+skip+pin, α=0.25 hotmap) GSM8K 100문항 + C32 재확인
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
until grep -q "GSM100_DONE" $SP/gsm100.log 2>/dev/null; do sleep 30; done
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide034_final_gsm100_promptmap; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
docker exec -d sgl-kt bash -c "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_final.log 2>&1"
i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
[ $ok = 1 ] || { echo "BOOT_FAIL" | tee $B/RUN.log; echo "GSM100B_DONE"; exit 1; }
echo "boot ok ${i}s" | tee $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
python3 $SP/gsm_eval.py $B/gsm100.json 100 2>&1 | tail -1 | sed "s/^/PROMPTMAP GSM100: /" | tee -a $B/RUN.log
rf=/tmp/final_c32.log; docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 128 --max-concurrency 32 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/final_c32.log 2>/dev/null
echo "PROMPTMAP C32: $(grep -h 'Output token throughput' $B/final_c32.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/final_c32.log|awk '{print $NF}')  (앞선 661.4/35.9)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "GSM100B_DONE $B"
