#!/usr/bin/env bash
# IDE_038: decode 배치에도 AMX mat_mul 경로 (KT_AMX_MIN_QLEN=0). (A) 패치·빌드 → 마이크로벤치 (B) 서빙 hot-96 α CF+τ C32/C64/GSM40, N=8+skip C32
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
true
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide045_hugepage; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "patch already applied (guard-fixed)" | tee -a $B/RUN.log
echo "== build $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_045.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_045.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_045.log' || { echo "BUILD_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_045.log | head -20 | tee -a $B/RUN.log; echo "IDE045_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED' | tee -a $B/RUN.log
echo "BUILD_OK $(date +%H:%M:%S)" | tee -a $B/RUN.log
# (A) 마이크로벤치: vec 경로(기본) vs mat 경로(KT_AMX_MIN_QLEN=0)
docker cp $SP/mb480_rows2.py sgl-kt:/tmp/
for mode in base hp; do env=""; [ $mode = hp ] && env="KT_HUGEPAGE=1"
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_rows2.py /tmp/mb_rows_$mode.json > /tmp/mb_rows_$mode.log 2>&1"; docker cp sgl-kt:/tmp/mb_rows_$mode.json $B/ 2>/dev/null
  echo "== rows bench ($mode: base vs KT_HUGEPAGE=1)" | tee -a $B/RUN.log; python3 -c "
import json; d=json.load(open('$B/mb_rows_$mode.json'))
for r in d['results']: print(f\"  n_cold={r['n_cold']:2d} T={r['T']:3d} rows/expert={8*r['T']/r['n_cold']:5.1f} us={r['us']:8.1f} us/expert={r['us_per_expert']:7.1f} us/row={r['us']/(8*r['T']):5.1f}\")" | tee -a $B/RUN.log; done
# (B) 서빙
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8"
boot() { docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "$2 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON > /tmp/sgl_045_$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A10 -E "Traceback|Error" /tmp/sgl_045_$1.log | tail -20 >> $B/RUN.log; return 1; }
bench() { local rf=/tmp/i45_$1.log; local NP=$(( $2==32 ? 128 : 192 ))
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2: $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}')  $3" | tee -a $B/RUN.log; }
COMMON="${COMMON/--mem-fraction-static 0.92 --max-total-tokens 49152/--mem-fraction-static 0.94 --max-total-tokens 110592 --kv-cache-dtype fp8_e5m2}"; COMMON="${COMMON/--cuda-graph-max-bs 64/--cuda-graph-max-bs 160}"
boot hp "KT_HUGEPAGE=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25" && { docker exec sgl-kt bash -c "grep AnonHugePages /proc/meminfo" | tee -a $B/RUN.log
  bench hp_c160 160 "(기준 982.4/119.5)"; bench hp_c64 64 "(기준 769/60.9)"; python3 $SP/gsm_eval.py $B/gsm40_hp.json 40 2>&1 | tail -1 | sed "s/^/hp GSM40: /" | tee -a $B/RUN.log; }
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "IDE045_DONE $B"
