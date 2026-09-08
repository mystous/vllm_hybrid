#!/usr/bin/env bash
# IDE_033 체인: hot-80 프로파일 종료 대기 → C++/Python 패치 적용 → wheel 빌드 → .so 설치 → hot-96 N=8 callback-free kill test → v3 셀 재개
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
until grep -q "H80_PROFILE_DONE\|BOOT_FAIL" $SP/h80prof.log 2>/dev/null; do sleep 30; done
echo "== IDE_033 build $(date +%H:%M:%S) =="
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
docker cp $SP/cf_cpp_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/cf_cpp_patch.py || { echo "CPP_PATCH_FAIL"; exit 1; }
docker cp $SP/cf_py_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/cf_py_patch.py || { echo "PY_PATCH_FAIL"; exit 1; }
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_cf.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_cf.log; tail -1 /tmp/ktbuild_cf.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_cf.log' || { echo "BUILD_FAIL"; docker exec sgl-kt grep -m3 -B2 -A3 "error:" /tmp/ktbuild_cf.log | head -20; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && python3 -c "from kt_kernel import kt_kernel_ext as k; ci=k.CPUInfer(2); print(\"cf bindings:\", all(hasattr(ci,n) for n in [\"arm_packet\",\"go_on_stream\",\"wait_done_on_stream\",\"rearm_packet\"]))" 2>&1 | grep -E "cf bindings|Error"'
echo "== IDE_033 kill test: hot-96 N=8 KT_CALLBACK_FREE=1 $(date +%H:%M:%S) =="
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$HOME/projects/vllm_hybrid/eval/results/$(date +%Y%m%d_%H%M%S)_ide033_cf_kill; mkdir -p $B
docker exec -d sgl-kt bash -c "KT_CALLBACK_FREE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_cf.log 2>&1"
i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
[ $ok = 1 ] || { echo "CF BOOT_FAIL" | tee $B/RUN.log; docker exec sgl-kt grep -n -m2 -B2 -A12 -E "Traceback|kt-cf|Error" /tmp/sgl_cf.log | tail -30 >> $B/RUN.log; exit 1; }
echo "CF boot ok ${i}s" | tee $B/RUN.log; docker exec sgl-kt grep -m2 "kt-cf" /tmp/sgl_cf.log | tee -a $B/RUN.log
for P in "The capital of France is" "def fibonacci(n):\n    "; do curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"q480\",\"prompt\":\"$P\",\"max_tokens\":24,\"temperature\":0}" | python3 -c "import json,sys; print('  greedy:', repr(json.load(sys.stdin)['choices'][0]['text'][:70]))" | tee -a $B/RUN.log; done
python3 - <<'PY' | tee -a $B/RUN.log
import json,urllib.request,concurrent.futures,time
def one(i):
    b=json.dumps({"model":"q480","prompt":"Write a haiku about number %d."%i,"max_tokens":48,"temperature":0}).encode()
    try: json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/v1/completions",b,{"Content-Type":"application/json"}),timeout=240)); return "ok"
    except Exception as e: return str(e)[:40]
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(32) as ex: rs=list(ex.map(one,range(32)))
print("BURST %d/32 in %.1fs"%(rs.count("ok"),time.time()-t0))
PY
grep -q "BURST 32/32" $B/RUN.log || { echo "CF BURST_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m3 -E "kt-cf|Traceback|Segfault" /tmp/sgl_cf.log | tail -5 >> $B/RUN.log; }
rf=/tmp/cf_c32.log; docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 128 --max-concurrency 32 --request-rate inf --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/cf_c32.log 2>/dev/null
echo "CF hot-96 N=8 C32: $(grep -h 'Output token throughput' $B/cf_c32.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/cf_c32.log|awk '{print $NF}')  (기준 505.4 / 48.3, kill ≥580)" | tee -a $B/RUN.log
python3 $SP/gsm_eval.py $B/cf_gsm40.json 40 2>&1 | tail -1 | sed "s/^/CF GSM40: /" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "CF_KILL_DONE $B"
# v3 셀 재개
sed -i 's|until grep -q "[^"]*" $SP/[a-z0-9_]*\.log 2>/dev/null; do sleep 30; done|true|' $SP/run_v3cells.sh
nohup bash $SP/run_v3cells.sh > $SP/v3cells.log 2>&1 &
echo "v3 relaunched"
