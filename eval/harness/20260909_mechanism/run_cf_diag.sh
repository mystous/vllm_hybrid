#!/usr/bin/env bash
# IDE_033 진단 체인: TaskQueue 계측 패치 → 빌드 → hot-96 N=8 CF (프로파일 + 계측) → hot-80 N=8 CF (계측) → hot-80 N=8 비-CF (계측) → v3 셀 재개
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
echo "== tq timing build $(date +%H:%M:%S) =="
docker cp $SP/tq_timing_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/tq_timing_patch.py || { echo "TQ_PATCH_FAIL"; exit 1; }
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_tq.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_tq.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_tq.log' || { echo "BUILD_FAIL"; docker exec sgl-kt grep -m3 -B2 -A3 "error:" /tmp/ktbuild_tq.log | head -20; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED'
echo "BUILD_OK $(date +%H:%M:%S)"
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide033_cf_diag; mkdir -p $B
boot() { # tag H KVT N envs
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; rm -rf /tmp/sgl_prof4; mkdir -p /tmp/sgl_prof4; true'; sleep 5
  docker exec -d sgl-kt bash -c "$5 KT_TQ_TIMING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens $3 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $2 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token $4 > /tmp/sgl_$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A12 -E "Traceback|kt-cf|Error" /tmp/sgl_$1.log | tail -30 >> $B/RUN.log; return 1; }
run() { # tag profile(0/1)
  local rf=/tmp/diag_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 160 --max-concurrency 32 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  if [ "$2" = 1 ]; then sleep 45; curl -s -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' -d '{"output_dir":"/tmp/sgl_prof4","num_steps":6,"activities":["CPU","GPU"]}' >/dev/null; sleep 15; curl -s -X POST http://127.0.0.1:30000/stop_profile >/dev/null; fi
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1_c32.log 2>/dev/null
  echo "$1 C32: $(grep -h 'Output token throughput' $B/$1_c32.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1_c32.log|awk '{print $NF}')" | tee -a $B/RUN.log
  docker exec sgl-kt grep "kt-tq" /tmp/sgl_$1.log > $B/$1_tq.txt; echo "  tq lines: $(wc -l < $B/$1_tq.txt); last: $(tail -1 $B/$1_tq.txt | cut -c1-230)" | tee -a $B/RUN.log
  if [ "$2" = 1 ]; then mkdir -p $B/$1_prof; docker cp sgl-kt:/tmp/sgl_prof4/. $B/$1_prof/ 2>/dev/null; python3 $SP/analyze_trace.py $B/$1_prof 2>&1 | tail -6 | tee -a $B/RUN.log; fi; }
# 1) hot-96 N=8 CF: 프로파일 + 계측
boot h96cf 96 49152 8 "KT_CALLBACK_FREE=1" && run h96cf 1
# 2) hot-80 N=8 CF: 계측
boot h80cf 80 131072 8 "KT_CALLBACK_FREE=1" && run h80cf 0
# 3) hot-80 N=8 비-CF (기존 경로): 계측 — def 소요시간 직접 대조
boot h80n8 80 131072 8 "" && run h80n8 0
# 4) hot-80 N=0 (immediate 전량): 계측 — immediate 경로 층당 시간
boot h80n0 80 131072 0 "" && run h80n0 0
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "CF_DIAG_DONE $B"
nohup bash $SP/run_v3cells.sh > $SP/v3cells.log 2>&1 &
echo "v3 relaunched"
