#!/usr/bin/env bash
# in-situ CPU 층 작업 phase 분해: A/B 종료 대기 → phase 프로파일 빌드 → hot-96 N=8 CF+skip (pinned) C32 → hot-80 N=8 동일 → 요약 → v3 재개
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
true
echo "== phase prof build $(date +%H:%M:%S) =="
docker cp $SP/phase_prof_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/phase_prof_patch.py || { echo "PATCH_FAIL"; echo "PHASE_PROF_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_pp.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_pp.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_pp.log' || { echo "BUILD_FAIL"; docker exec sgl-kt grep -m3 -B2 -A3 "error:" /tmp/ktbuild_pp.log | head -20; echo "PHASE_PROF_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED'
echo "BUILD_OK $(date +%H:%M:%S)"
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide033_phase_prof; mkdir -p $B; docker cp $SP/pin_nonkt.sh sgl-kt:/tmp/pin_nonkt.sh
one() { # tag H KVT
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "KT_PHASE_PROF=1 KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_TQ_TIMING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens $3 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts $2 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_pp_$1.log 2>&1"
  local i=0 ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  [ $ok = 1 ] || { echo "$1 BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B2 -A12 -E "Traceback|Error" /tmp/sgl_pp_$1.log | tail -20 >> $B/RUN.log; return 1; }
  echo "$1 boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "grep -c 'Profiling Results' /tmp/sgl_pp_$1.log" > /dev/null; local rf=/tmp/pp_$1.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 128 --max-concurrency 32 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1_c32.log 2>/dev/null
  echo "$1 C32: $(grep -h 'Output token throughput' $B/$1_c32.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1_c32.log|awk '{print $NF}')" | tee -a $B/RUN.log
  # C64 재현 (hot-96 skip+pinned C64 760.1 확인용; hot-80 은 hot-80 N=4 C64 487 대조)
  rf=/tmp/pp_$1_c64.log
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 192 --max-concurrency 64 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/$1_c64.log 2>/dev/null
  echo "$1 C64: $(grep -h 'Output token throughput' $B/$1_c64.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1_c64.log|awk '{print $NF}')" | tee -a $B/RUN.log
  docker exec sgl-kt grep "Profiling Results" /tmp/sgl_pp_$1.log > $B/$1_phases.txt; echo "  phase lines: $(wc -l < $B/$1_phases.txt)" | tee -a $B/RUN.log
  python3 - $B/$1_phases.txt <<'PY' | tee -a $B/RUN.log
import re,sys,statistics as st
rows=[]
for l in open(sys.argv[1]):
    m=re.search(r"numa\[(\d)\]\): activated_expert: (\d+), prepare: (\d+) us, cpy_input: (\d+) us, q_input: (\d+) us, up_gate: (\d+) us, act: (\d+) us, q_down: (\d+) us, down: (\d+) us, weight: (\d+) us, total: (\d+) us, max_local_num: (\d+), qlen: (\d+)",l)
    if m: rows.append([int(x) for x in m.groups()])
names=["numa","act_exp","prepare","cpy_input","q_input","up_gate","act","q_down","down","weight","total","max_local","qlen"]
dec=[r for r in rows if 16<=r[12]<=40]
print(f"  rows={len(rows)} decode(qlen 16..40)={len(dec)}")
for numa in (0,1):
    d=[r for r in dec if r[0]==numa]
    if not d: continue
    med=[st.median([r[i] for r in d]) for i in range(len(names))]
    print("  numa%d median: "%numa + " ".join(f"{names[i]}={med[i]:.0f}" for i in range(1,13)))
    # 활성 expert 수별 total 중앙값
    by={}
    for r in d: by.setdefault(r[1],[]).append(r[10])
    print("  numa%d total by act_exp: "%numa + " ".join(f"{k}:{st.median(v):.0f}(n{len(v)})" for k,v in sorted(by.items())[:12]))
PY
}
one h96 96 49152
one h80 80 131072
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "PHASE_PROF_DONE $B"
