#!/usr/bin/env bash
# decode 라우팅 트레이스 → hotmap_decode.json → hot-96 N=8 (CF+skip+pin) 재측정. 병행: wrapper 타이밍 빌드. 종료 후 v3 재개.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "PHASE_PROF_DONE" $SP/phaseprof.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide034_decode_hotmap; mkdir -p $B; docker cp $SP/pin_nonkt.sh sgl-kt:/tmp/pin_nonkt.sh
COMMON="--served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8"
ENVS="KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_TQ_TIMING=1 CUDA_VISIBLE_DEVICES=0,1,2,3"
boot() { # tag hotmap extra
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "$ENVS $4 python3 -m sglang.launch_server --model-path $M $COMMON --init-expert-location $2 $3 > /tmp/sgl_$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A12 -E "Traceback|Error" /tmp/sgl_$1.log | tail -20 >> $B/RUN.log; return 1; }
bench() { # tag C nprompts in out note
  local rf=/tmp/dh_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len $4 --sonnet-output-len $5 --sonnet-prefix-len $(( $4 < 200 ? 30 : 100 )) --num-prompts $3 --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2 in$4/out$5: $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}')  $6" | tee -a $B/RUN.log; }
# 1) decode 트레이스 수집 (현재 hotmap, recorder). 워크로드: 입력 64 / 출력 512 (decode 89%)
docker exec sgl-kt bash -c 'rm -f /tmp/expert_distribution_recorder_*.pt; true'
boot trace /tmp/hotmap.json "--expert-distribution-recorder-mode per_pass" "" || { echo "DECODE_HOTMAP_DONE"; exit 1; }
curl -s -X POST http://127.0.0.1:30000/start_expert_distribution_record >/dev/null; echo "record start" | tee -a $B/RUN.log
# 병행: wrapper 타이밍 빌드 (측정 민감하지 않은 트레이스 구간)
docker cp $SP/wrapper_timing_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/wrapper_timing_patch.py | tee -a $B/RUN.log
docker exec -d sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_wp.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_wp.log'
bench trace_in64_out512 32 96 64 512 "(트레이스 수집용)"
curl -s -X POST http://127.0.0.1:30000/stop_expert_distribution_record >/dev/null; curl -s -X POST http://127.0.0.1:30000/dump_expert_distribution_record >/dev/null; sleep 5
docker cp $SP/build_hotmap_decode.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/build_hotmap_decode.py 2>&1 | tee -a $B/RUN.log
docker cp sgl-kt:/tmp/hotmap_decode.json $B/ 2>/dev/null
# 빌드 종료 대기 → 설치
w=0; until docker exec sgl-kt bash -c 'grep -q BUILD_EXIT /tmp/ktbuild_wp.log' 2>/dev/null; do sleep 15; w=$((w+15)); ((w>900)) && break; done
if docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_wp.log'; then docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo WRAP_BUILD_INSTALLED' | tee -a $B/RUN.log; else echo "WRAP_BUILD_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A4 "error:" /tmp/ktbuild_wp.log | head -14 >> $B/RUN.log; fi
# 2) decode hotmap 으로 재측정 (phase prof + wrapper 타이밍 켜서)
boot dec /tmp/hotmap_decode.json "" "KT_PHASE_PROF=1" || { echo "DECODE_HOTMAP_DONE"; exit 1; }
for P in "The capital of France is" "def fibonacci(n):\n    "; do curl -s http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"q480\",\"prompt\":\"$P\",\"max_tokens\":24,\"temperature\":0}" | python3 -c "import json,sys; print('  greedy:', repr(json.load(sys.stdin)['choices'][0]['text'][:70]))" | tee -a $B/RUN.log; done
bench dec_c32 32 128 512 128 "(prompt-hotmap 기준 600/43.5)"
bench dec_c64 64 192 512 128 "(prompt-hotmap 기준 760~786)"
python3 $SP/gsm_eval.py $B/gsm40.json 40 2>&1 | tail -1 | sed "s/^/DEC GSM40: /" | tee -a $B/RUN.log
docker exec sgl-kt grep "Profiling Results" /tmp/sgl_dec.log > $B/dec_phases.txt; docker exec sgl-kt grep "kt-wrap" /tmp/sgl_dec.log > $B/dec_wrap.txt
python3 - $B/dec_phases.txt $B/dec_wrap.txt <<'PY' | tee -a $B/RUN.log
import re,sys,statistics as st
rows=[]
for l in open(sys.argv[1]):
    m=re.search(r"numa\[(\d)\]\): activated_expert: (\d+), prepare: (\d+) us, cpy_input: (\d+) us, q_input: (\d+) us, up_gate: (\d+) us, act: (\d+) us, q_down: (\d+) us, down: (\d+) us, weight: (\d+) us, total: (\d+) us, max_local_num: (\d+), qlen: (\d+)",l)
    if m: rows.append([int(x) for x in m.groups()])
dec=[r for r in rows if 16<=r[12]<=40 and r[0]==0]
if dec: print(f"  decode(qlen16..40,numa0) n={len(dec)} act_exp median={st.median([r[1] for r in dec])} mean={st.mean([r[1] for r in dec]):.2f} total median={st.median([r[10] for r in dec])}")
w=[]
for l in open(sys.argv[2]):
    m=re.search(r"qlen: (\d+), numa_job: (\d+) us, merge: (\d+) us, incremental: (\d)",l)
    if m: w.append([int(x) for x in m.groups()])
wd=[r for r in w if 16<=r[0]<=40]
if wd: print(f"  wrap decode n={len(wd)} numa_job median={st.median([r[1] for r in wd])} merge median={st.median([r[2] for r in wd])} p90 merge={sorted([r[2] for r in wd])[int(0.9*len(wd))]}")
PY
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "DECODE_HOTMAP_DONE $B"
