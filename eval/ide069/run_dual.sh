#!/usr/bin/env bash
# IDE_069 / TSK_053 — TP4+CPU 오프로딩 ×2 (GPU 8장) 대 GPU-only TP8 성능 비교.
#
# 인스턴스 A = GPU 0-3 + 소켓0,  B = GPU 4-7 + 소켓1.  8-29 dual 실험에서 kt-kernel 스레드가
# numactl 을 무시하고 절대 0번 코어부터 고정되는 결함이 확인됐으므로, 인스턴스마다 **cgroup
# cpuset 컨테이너**(소켓 전체 코어 + 메모리 노드)에 가둔다. 두 컨테이너는 현재 sgl-kt 를 그대로
# 이미지로 커밋해 만들므로 소프트웨어 상태가 동일하다.
# 오프로딩 두 변형: (1) expert 전량 CPU  (2) hot96 + hotmap + deferral 4 + cuda graph.
# 기준선: GPU-only TP8+EP8 (sgl-kt, cpuset 없음) — C32 / C64.
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide069_dual_vs_tp8
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
# nerdctl commit 이 snapshot mount 오류로 실패(15:01) → 8-30 dual 에서 만든 cpuset 컨테이너를 재사용한다.
# sgl-kt5 = 소켓0 (0-55,112-167 / mems 0), sgl-kt2 = 소켓1 (56-111,168-223 / mems 1). kt-kernel 0.7.0.post2 + 패치.
S0=sgl-kt5; S1=sgl-kt2
SKIP_REF=${SKIP_REF:-0}
KTC="--kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2"
HOT="--kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic"
COLD="--kt-num-gpu-experts 0 --disable-cuda-graph --mem-fraction-static 0.80 --max-total-tokens 65536"
log "== IDE_069 dual vs tp8 start $TS =="
log "turbo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"

# ---------- 0. 컨테이너 상태 확인 ----------
stop_server
for c in $S0 $S1; do docker start $c >/dev/null 2>&1; docker exec $c bash -c "grep Cpus_allowed_list /proc/self/status; pip show kt-kernel | grep ^Version" 2>&1 | tr "\n" " " | tee -a "$RUN_LOG"; echo | tee -a "$RUN_LOG"; done

# ---------- 공용 ----------
boot_in() {  # boot_in <cn> <port> <gpus> <out> <args...>
  local cn=$1 port=$2 gpus=$3 out=$4; shift 4
  mkdir -p "$out"
  echo "CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server $*" > "$out/launch_cmd.txt"
  docker exec -d "$cn" bash -c "CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server $* > /tmp/sgl_server.log 2>&1"
}
wait_health() {  # wait_health <cn> <port> <timeout>  → 0 ok
  local cn=$1 port=$2 tmo=$3 i=0
  while ((i<tmo)); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    docker exec "$cn" pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || return 1
    sleep 10; i=$((i+10))
  done; return 2
}
stop_all() { for c in sgl-kt $S0 $S1; do docker exec $c bash -c 'pkill -f sglang.launch_serve[r] 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true' >/dev/null 2>&1; done; sleep 5; }
smoke_port() {  # smoke_port <port> <out>
  local port=$1 out=$2; : > "$out/smoke.json"
  for p in "The capital of France is" "def fibonacci(n):" "1+2+3+...+100 ="; do
    curl -s -m 600 "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"prompt\":\"$p\",\"max_tokens\":48,\"temperature\":0}" >> "$out/smoke.json"; echo >> "$out/smoke.json"; done
  curl -s -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that reverses a string. Just the code.\"}],\"max_tokens\":96,\"temperature\":0}" > "$out/smoke_chat.json"
  python3 - "$out" <<'PY'
import json,sys,os
out=sys.argv[1]; t=[]
for l in open(os.path.join(out,'smoke.json')):
    l=l.strip()
    if l:
        try: t.append(json.loads(l)['choices'][0]['text'])
        except Exception as e: t.append(f'<err {e}>')
try: t.append(json.load(open(os.path.join(out,'smoke_chat.json')))['choices'][0]['message']['content'])
except Exception as e: t.append(f'<err {e}>')
open(os.path.join(out,'smoke_texts.txt'),'w').write(''.join(f'--- Q{i} ---\n{x}\n\n' for i,x in enumerate(t,1)))
print(' | '.join(x[:40].replace('\n','⏎') for x in t))
PY
}
bench_port() {  # bench_port <port> <out> <conc> <n>   (백그라운드에서 호출)
  local port=$1 out=$2 conc=$3 n=$4; mkdir -p "$out"
  docker exec "$BENCH_CN" vllm bench serve --backend openai --base-url "http://127.0.0.1:$port" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer "$TOK" --dataset-name sonnet --dataset-path /tmp/sonnet.txt \
    --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 \
    --num-prompts "$n" --max-concurrency "$conc" --request-rate inf --seed 42 \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > "$out/bench.log" 2>&1
  grep -E "Successful requests|Benchmark duration|Output token throughput|Total token throughput|Median TTFT|P95 TTFT|Median TPOT|P95 TPOT" "$out/bench.log" | sed 's/  */ /g' > "$out/bench_summary.txt"
}
samplers_on()  { mkdir -p "$1"; ( nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > "$1/gpu_util.csv" 2>&1 & echo $! > "$1/.gpu_mon" ); ( bash "$CPU_SAMPLER" "$1/cpu_util.txt" & echo $! > "$1/.cpu_mon" ); }
samplers_off() { kill "$(cat "$1/.gpu_mon")" 2>/dev/null; kill "$(cat "$1/.cpu_mon")" 2>/dev/null; }
sum_tps() { python3 - "$@" <<'PY'
import sys,re
tot=0.0
for f in sys.argv[1:]:
    for l in open(f):
        m=re.search(r"Output token throughput \(tok/s\): ([0-9.]+)", l)
        if m: tot+=float(m.group(1))
print(f"{tot:.2f}")
PY
}

# ---------- 1. 기준선: GPU-only TP8+EP8 (C32, C64) ----------
if [ "$SKIP_REF" = 0 ]; then
log "-- ref: GPU-only TP8+EP8 --"
out=$BASE/ref_gpu_tp8_ep8
boot_in sgl-kt $PORT 0,1,2,3,4,5,6,7 "$out" --model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 8 --ep-size 8 --attention-backend triton --trust-remote-code --mem-fraction-static 0.90 --max-total-tokens 131072
wait_health sgl-kt $PORT 2400; rc=$?; log "ref boot rc=$rc"; echo "rc=$rc" > "$out/verdict.txt"
if [ $rc = 0 ]; then
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/hbm_after_boot.csv"
  smoke_port $PORT "$out" | tee -a "$RUN_LOG"
  for c in 32 64; do samplers_on "$out/c$c"; bench_port $PORT "$out/c$c" $c $((c*4)); samplers_off "$out/c$c"; log "ref C$c: $(cat $out/c$c/bench_summary.txt | tr '\n' ' ')"; done
fi
stop_all
fi

# ---------- 2. dual ----------
run_dual() {  # run_dual <name> <extra-args>
  local name=$1; shift
  local out=$BASE/$name; mkdir -p "$out/A" "$out/B"
  log "-- dual $name --"
  free -g > "$out/free_before.txt"
  boot_in $S0 30000 0,1,2,3 "$out/A" --model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 $KTC --kt-cpuinfer 48 "$@"
  boot_in $S1 30001 4,5,6,7 "$out/B" --model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port 30001 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 $KTC --kt-cpuinfer 48 "$@"
  wait_health $S0 30000 2400; ra=$?; wait_health $S1 30001 2400; rb=$?
  log "boot A rc=$ra B rc=$rb"; echo "A=$ra B=$rb" > "$out/verdict.txt"
  docker exec $S0 bash -c 'tail -60 /tmp/sgl_server.log' > "$out/A/server_tail.log" 2>/dev/null
  docker exec $S1 bash -c 'tail -60 /tmp/sgl_server.log' > "$out/B/server_tail.log" 2>/dev/null
  if [ $ra = 0 ] && [ $rb = 0 ]; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/hbm_after_boot.csv"; free -g > "$out/free_after_boot.txt"
    log "smoke A: $(smoke_port 30000 "$out/A")"; log "smoke B: $(smoke_port 30001 "$out/B")"
    for c in 16 32; do
      mkdir -p "$out/c$c"; samplers_on "$out/c$c"
      bench_port 30000 "$out/c$c/A" $c $((c*4)) & pa=$!
      bench_port 30001 "$out/c$c/B" $c $((c*4)) & pb=$!
      wait $pa; wait $pb; samplers_off "$out/c$c"
      log "dual $name C${c}each: A=$(grep -oE 'Output token throughput \(tok/s\): [0-9.]+' $out/c$c/A/bench_summary.txt | grep -oE '[0-9.]+$')  B=$(grep -oE 'Output token throughput \(tok/s\): [0-9.]+' $out/c$c/B/bench_summary.txt | grep -oE '[0-9.]+$')  합산=$(sum_tps $out/c$c/A/bench_summary.txt $out/c$c/B/bench_summary.txt)"
      cpu_summary "$out/c$c/cpu_util.txt" | tee -a "$RUN_LOG"
    done
  fi
  stop_all
}
# ---------- 1b. 검증: sgl-kt5 단독 TP4 hot96 def4 (sgl-kt 의 t2 519.8 과 대조 — 컨테이너 등가성) ----------
out=$BASE/single_s5_hot96_def4; mkdir -p "$out"
log "-- single in $S0 (hot96 def4, cpuinfer 96) --"
boot_in $S0 30000 0,1,2,3 "$out" --model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 $KTC --kt-cpuinfer 96 $HOT
wait_health $S0 30000 2400; rc=$?; log "single boot rc=$rc"; echo "rc=$rc" > "$out/verdict.txt"
if [ $rc = 0 ]; then
  log "smoke: $(smoke_port 30000 "$out")"
  samplers_on "$out/c32"; bench_port 30000 "$out/c32" 32 128; samplers_off "$out/c32"; log "single C32: $(cat $out/c32/bench_summary.txt | tr '\n' ' ')"
else docker exec $S0 bash -c 'tail -40 /tmp/sgl_server.log' > "$out/server_tail.log"; fi
stop_all

run_dual dual_cold_expert0 $COLD
run_dual dual_hot96_def4  $HOT
log "== dual done $(date +%H:%M:%S) =="
echo "$BASE"
