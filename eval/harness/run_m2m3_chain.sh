#!/usr/bin/env bash
# M2 (HiCache 공유prefix 이득/간섭) → M3 준비 (235B kt INT4 변환). 마커 대기 없음.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$HOME/projects/vllm_hybrid/eval/results/$(date +%Y%m%d_%H%M%S)_pln008_m2_hicache_prefix; mkdir -p $B
boot() { docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 4 $1 > /tmp/sgl_m2.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && return 0; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || return 1; sleep 10; i=$((i+10)); done; return 1; }
bench() { local rf=/tmp/m2_$1.log
  docker exec -d vllm-h100 bash -c "timeout 400 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name random --random-input-len 1600 --random-output-len 256 --random-prefix-len 1500 --num-prompts 32 --max-concurrency 32 --request-rate inf --seed $2 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>460)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1: TTFT $(grep -h 'Mean TTFT' $B/$1.log|awk '{print $NF}') TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}') tput $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}')" | tee -a $B/RUN.log; }
echo "== M2 HiCache 공유prefix (in1600 prefix1500 out256 C32) $(date +%H:%M:%S) ==" | tee $B/RUN.log
for mode in gpu hicache; do
  extra=""; [ $mode = hicache ] && extra="--enable-hierarchical-cache --hicache-size 64"
  boot "$extra" && { echo "[$mode] boot ok" | tee -a $B/RUN.log; bench ${mode}_w1 42; bench ${mode}_w2 42; } || echo "[$mode] BOOT_FAIL" | tee -a $B/RUN.log
done
echo "M2_DONE $B" | tee -a $B/RUN.log
echo "== M3 준비: 235B kt INT4 변환 $(date +%H:%M:%S) ==" | tee -a $B/RUN.log
SNAP2=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/*/ | head -1)
M2P=/models/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/$(basename "$SNAP2")
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 3
docker exec sgl-kt bash -c "cd /sgl-workspace/ktransformers/kt-kernel/scripts && python3 convert_cpu_weights.py --input-path $M2P --input-type fp8 --output /models/kt/qwen3-235b-int4 --quant-method int4 --cpuinfer-threads 96 --threadpool-count 2 > /tmp/convert_235b.log 2>&1; echo CONVERT_EXIT=\$? >> /tmp/convert_235b.log; tail -2 /tmp/convert_235b.log; du -sh /models/kt/qwen3-235b-int4 2>/dev/null"
echo "CONVERT_235B_DONE $(date +%H:%M:%S)" | tee -a $B/RUN.log
