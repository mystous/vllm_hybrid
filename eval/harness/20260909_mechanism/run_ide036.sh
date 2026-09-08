#!/usr/bin/env bash
# IDE_036: 폴러 수준 빈 deferred 생략 — τ 스윕 종료 후 패치·빌드 → 235B hot-96 α (C32/C64/GSM40) → 480B hot-96 α C32 sanity
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "TAU_SWEEP_DONE" $SP/tau_sweep.log 2>/dev/null; do sleep 30; done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "== IDE_036 build $(date +%H:%M:%S) =="
docker cp $SP/cf_skipdef_cpp_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/cf_skipdef_cpp_patch.py || { echo "CPP_PATCH_FAIL"; echo "IDE036_DONE"; exit 1; }
docker cp $SP/cf_skipdef_py_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/cf_skipdef_py_patch.py || { echo "PY_PATCH_FAIL"; echo "IDE036_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_036.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_036.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_036.log' || { echo "BUILD_FAIL"; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_036.log | head -24; echo "IDE036_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && python3 -c "from kt_kernel import kt_kernel_ext as k; print(\"cf_def_stats:\", hasattr(k.CPUInfer(2), \"cf_def_stats\"))"'
echo "BUILD_OK $(date +%H:%M:%S)"
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide036_skipdef; mkdir -p $B
boot() { # tag model_path served common kt_args envs
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$5 python3 -m sglang.launch_server --model-path $2 --served-model-name $3 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --cuda-graph-max-bs 64 $4 > /tmp/sgl_036_$1.log 2>&1"
  local i=0; while ((i<1800)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A12 -E "Traceback|Error|kt-cf" /tmp/sgl_036_$1.log | tail -24 >> $B/RUN.log; return 1; }
bench() { # tag served tok C nprompts note
  local rf=/tmp/i36_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model $2 --tokenizer $3 --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $5 --max-concurrency $4 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$4: tok/s $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}') TTFT $(grep -h 'Mean TTFT' $B/$1.log|awk '{print $NF}')  $6" | tee -a $B/RUN.log; }
ENVS="KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_CF_SKIP_EMPTY_DEF=1 CUDA_VISIBLE_DEVICES=0,1,2,3"
# 235B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/$(basename "$SNAP")
KT="--mem-fraction-static 0.90 --max-total-tokens 131072 --kt-weight-path /models/kt/qwen3-235b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --kt-num-gpu-experts 96 --init-expert-location /tmp/hm235/hotmap_alpha0.25.json"
boot q235 $M q235 "$KT" "$ENVS" && { for P in "The capital of France is"; do curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"q235\",\"prompt\":\"$P\",\"max_tokens\":24,\"temperature\":0}" | python3 -c "import json,sys; print('  greedy:', repr(json.load(sys.stdin)['choices'][0]['text'][:70]))" | tee -a $B/RUN.log; done
  bench q235_c32 q235 $TOK 32 128 "(235B hybrid α 기준 804/28.7, GPU-only 930/21.2)"; bench q235_c64 q235 $TOK 64 192 "(기준 1549/32.9, GPU-only 1916/24.8)"
  python3 $SP/gsm_eval.py $B/gsm40_235.json 40 q235 2>&1 | tail -1 | sed "s/^/q235 GSM40: /" | tee -a $B/RUN.log; }
# 480B sanity
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
KT="--mem-fraction-static 0.92 --max-total-tokens 49152 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json"
boot q480 $M q480 "$KT" "$ENVS" && { bench q480_c32 q480 $TOK 32 128 "(480B α 기준 659~661/36)"; python3 $SP/gsm_eval.py $B/gsm40_480.json 40 2>&1 | tail -1 | sed "s/^/q480 GSM40: /" | tee -a $B/RUN.log; }
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "IDE036_DONE $B"
