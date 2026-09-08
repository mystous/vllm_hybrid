#!/usr/bin/env bash
# IDE_033-b: 빈 immediate 생략 (KT_CF_SKIP_EMPTY_IMM=1) hot-96 N=8 — 진단 체인 종료 후 실행. greedy·버스트·C32·GSM40·tq 계측
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "CF_DIAG_DONE\|FAIL" $SP/cf_diag.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide033b_skipimm_pin; mkdir -p $B; docker cp $SP/pin_nonkt.sh sgl-kt:/tmp/pin_nonkt.sh
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
docker exec -d sgl-kt bash -c "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_TQ_TIMING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_cfskip.log 2>&1"
i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
[ $ok = 1 ] || { echo "SKIP BOOT_FAIL" | tee $B/RUN.log; docker exec sgl-kt grep -n -m2 -B2 -A12 -E "Traceback|kt-cf|Error" /tmp/sgl_cfskip.log | tail -30 >> $B/RUN.log; echo "SKIP_TEST_DONE"; exit 1; }
echo "SKIP boot ok ${i}s" | tee $B/RUN.log
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
bench() { # tag C nprompts note
  local rf=/tmp/cfskip_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $3 --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2: $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}')  $4" | tee -a $B/RUN.log
  docker exec sgl-kt grep "kt-tq" /tmp/sgl_cfskip.log | tail -1 | cut -c1-230 | sed "s/^/  tq: /" | tee -a $B/RUN.log; }
bench h96_skip_unpinned 32 128 "(CF 531.0/45.5, 기준 505.4/48.3, kill ≥580)"
docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
bench h96_skip_pinned 32 128 "(pinned)"
bench h96_skip_pinned_r2 32 128 "(pinned, 반복)"
python3 $SP/gsm_eval.py $B/gsm40.json 40 2>&1 | tail -1 | sed "s/^/SKIP GSM40: /" | tee -a $B/RUN.log
bench h96_skip_pinned_c64 64 192 "(pinned; hot-96+def4 C64 408.2)"
# hot-80 N=8 CF+SKIP, KV 131072: unpinned → pinned (hot-80 체제 = CPU 한계, def 시간 자체가 관건)
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
docker exec -d sgl-kt bash -c "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_TQ_TIMING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 131072 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 80 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_cfskip.log 2>&1"
i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
if [ $ok = 1 ]; then echo "h80 SKIP boot ok ${i}s" | tee -a $B/RUN.log
  bench h80_skip_unpinned 32 128 "(hot-80 N=8 기준 343.4/76.8)"
  docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
  bench h80_skip_pinned 32 128 "(pinned)"
  bench h80_skip_pinned_c64 64 192 "(pinned; hot-80 N=4 C64 487.0)"
else echo "h80 SKIP BOOT_FAIL" | tee -a $B/RUN.log; fi
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "SKIP_TEST_DONE $B"
