#!/usr/bin/env bash
# IDE_069 / TSK_053 확장 — dual 인스턴스 앞에 sglang_router 를 두고 **단일 엔드포인트**로 벤치.
# 합산 방식(run_dual.sh)과의 차이 = 라우터 오버헤드 + 부하 불균형. 두 오프로딩 변형(expert 0 / hot96 def4) 모두.
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide069_router_dual${TAG}
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480; S0=sgl-kt5; S1=sgl-kt2; RPORT=30002
ONLY_HOT=${ONLY_HOT:-0}
TAG=${TAG:-}
KTC="--kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2"
HOT="--kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic"
COLD="--kt-num-gpu-experts 0 --disable-cuda-graph --mem-fraction-static 0.80 --max-total-tokens 65536"
log "== IDE_069 router-dual start $TS =="
boot_in() { local cn=$1 port=$2 gpus=$3 out=$4; shift 4; mkdir -p "$out"; echo "CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server $*" > "$out/launch_cmd.txt"; docker exec -d "$cn" bash -c "CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server $* > /tmp/sgl_server.log 2>&1"; }
wait_health() { local cn=$1 port=$2 tmo=$3 i=0; while ((i<tmo)); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0; docker exec "$cn" pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || return 1; sleep 10; i=$((i+10)); done; return 2; }
stop_all() { for c in sgl-kt $S0 $S1; do docker exec $c bash -c 'pkill -f sglang_router.launch_route[r] 2>/dev/null; pkill -f sglang.launch_serve[r] 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true' >/dev/null 2>&1; done; sleep 5; }
smoke_port() { local port=$1 out=$2; : > "$out/smoke.json"; for p in "The capital of France is" "def fibonacci(n):" "1+2+3+...+100 ="; do curl -s -m 600 "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"prompt\":\"$p\",\"max_tokens\":48,\"temperature\":0}" >> "$out/smoke.json"; echo >> "$out/smoke.json"; done; curl -s -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that reverses a string. Just the code.\"}],\"max_tokens\":96,\"temperature\":0}" > "$out/smoke_chat.json"; python3 - "$out" <<'PY'
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
bench_port() { local port=$1 out=$2 conc=$3 n=$4; mkdir -p "$out"; ( nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > "$out/gpu_util.csv" 2>&1 & echo $! > "$out/.gpu_mon" ); ( bash "$CPU_SAMPLER" "$out/cpu_util.txt" & echo $! > "$out/.cpu_mon" ); docker exec "$BENCH_CN" vllm bench serve --backend openai --base-url "http://127.0.0.1:$port" --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts "$n" --max-concurrency "$conc" --request-rate inf --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > "$out/bench.log" 2>&1; kill "$(cat "$out/.gpu_mon")" 2>/dev/null; kill "$(cat "$out/.cpu_mon")" 2>/dev/null; grep -E "Successful requests|Benchmark duration|Output token throughput|Total token throughput|Median TTFT|P95 TTFT|Median TPOT|P95 TPOT" "$out/bench.log" | sed 's/  */ /g' > "$out/bench_summary.txt"; }

run_variant() {  # run_variant <name> <policies(,)> <extra-args>
  local name=$1 pols=$2; shift 2
  local out=$BASE/$name; mkdir -p "$out/A" "$out/B"
  log "-- variant $name --"
  boot_in $S0 30000 0,1,2,3 "$out/A" --model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 $KTC --kt-cpuinfer 48 "$@"
  boot_in $S1 30001 4,5,6,7 "$out/B" --model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port 30001 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 $KTC --kt-cpuinfer 48 "$@"
  wait_health $S0 30000 2400; ra=$?; wait_health $S1 30001 2400; rb=$?
  log "boot A rc=$ra B rc=$rb"; echo "A=$ra B=$rb" > "$out/verdict.txt"
  if [ $ra = 0 ] && [ $rb = 0 ]; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/hbm_after_boot.csv"
    for pol in ${pols//,/ }; do
      docker exec -d $S0 bash -c "python3 -m sglang_router.launch_router --worker-urls http://127.0.0.1:30000 http://127.0.0.1:30001 --host 127.0.0.1 --port $RPORT --policy $pol > /tmp/router.log 2>&1"
      i=0; while ((i<60)); do curl -sf "http://127.0.0.1:$RPORT/health" >/dev/null 2>&1 && break; sleep 3; i=$((i+3)); done
      log "router($pol) up after ${i}s: $(curl -s -m5 http://127.0.0.1:$RPORT/health | head -c 80)"
      ro=$out/router_$pol; mkdir -p "$ro"
      log "smoke via router: $(smoke_port $RPORT "$ro")"
      for c in 32 64; do bench_port $RPORT "$ro/c$c" $c $((c*4)); log "router($pol) C$c: $(cat $ro/c$c/bench_summary.txt | tr '\n' ' ')"; cpu_summary "$ro/c$c/cpu_util.txt" | tee -a "$RUN_LOG"; done
      docker exec $S0 bash -c 'tail -30 /tmp/router.log' > "$ro/router_tail.log" 2>/dev/null
      docker exec $S0 bash -c 'pkill -f sglang_router.launch_route[r]; true' >/dev/null 2>&1; sleep 3
    done
    docker exec $S0 bash -c 'tail -60 /tmp/sgl_server.log' > "$out/A/server_tail.log" 2>/dev/null
    docker exec $S1 bash -c 'tail -60 /tmp/sgl_server.log' > "$out/B/server_tail.log" 2>/dev/null
  fi
  stop_all
}
stop_all
[ "$ONLY_HOT" = 1 ] || run_variant router_cold_expert0 round_robin $COLD
run_variant router_hot96_def4 round_robin,cache_aware $HOT
log "== router done $(date +%H:%M:%S) =="
echo "$BASE"
