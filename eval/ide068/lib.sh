#!/usr/bin/env bash
# IDE_068 공용 함수 — 서버 기동/대기/중지, 품질 smoke, 벤치, 샘플러.
# 선행 `run_tsk047.sh` 의 절차를 그대로 승계하되, 셀마다 DRAM·HBM 사용량을 추가 기록한다.
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
REPO=$HOME/projects/vllm_hybrid
CPU_SAMPLER=$REPO/eval/ide068/cpu_sample.sh
CN=${CN:-sgl-kt}                       # 서빙 컨테이너
BENCH_CN=${BENCH_CN:-vllm-h100}        # 벤치 클라이언트 컨테이너
PORT=${PORT:-30000}
LOG_IN_CN=/tmp/sgl_server.log

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$RUN_LOG"; }

# 컨테이너 안 sglang 서버 전부 종료
stop_server() {
  docker exec "$CN" bash -c 'pkill -f sglang.launch_serve[r] 2>/dev/null; sleep 6; pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true' >/dev/null 2>&1
  sleep 4
}

# boot <cell_dir> <CUDA_VISIBLE_DEVICES> <timeout_s> <server args...>
# 결과: 전역 BOOT_VERDICT ∈ {HEALTH_OK, DIED, TIMEOUT}, BOOT_SEC
boot() {
  local out=$1 gpus=$2 tmo=$3; shift 3
  mkdir -p "$out"
  free -g > "$out/free_before.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$out/hbm_before.csv"
  echo "CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server $*" > "$out/launch_cmd.txt"
  docker exec -d "$CN" bash -c "CUDA_VISIBLE_DEVICES=$gpus python3 -m sglang.launch_server $* > $LOG_IN_CN 2>&1"
  local i=0; BOOT_VERDICT=TIMEOUT
  while ((i<tmo)); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then BOOT_VERDICT=HEALTH_OK; break; fi
    docker exec "$CN" pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || { BOOT_VERDICT=DIED; break; }
    sleep 10; i=$((i+10))
  done
  BOOT_SEC=$i
  log "boot verdict=$BOOT_VERDICT ${BOOT_SEC}s gpus=$gpus"
  docker exec "$CN" bash -c "tail -200 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  docker exec "$CN" bash -c "grep -iE 'out of memory|OOM|CUDA error|Error|not divisible' $LOG_IN_CN | tail -5" > "$out/error_lines.log" 2>/dev/null
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    sleep 5
    free -g > "$out/free_after_boot.txt"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "$out/hbm_after_boot.csv"
  fi
}

# greedy 4문항 (선행 캠페인과 동일 프롬프트)
smoke() {
  local out=$1 model=$2
  : > "$out/smoke.json"
  for p in "The capital of France is" "def fibonacci(n):" "1+2+3+...+100 ="; do
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"prompt\":\"$p\",\"max_tokens\":48,\"temperature\":0}" >> "$out/smoke.json"
    echo >> "$out/smoke.json"
  done
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that reverses a string. Just the code.\"}],\"max_tokens\":96,\"temperature\":0}" > "$out/smoke_chat.json"
  python3 - "$out" <<'PY'
import json,sys,os
out=sys.argv[1]
texts=[]
for line in open(os.path.join(out,'smoke.json')):
    line=line.strip()
    if not line: continue
    try: texts.append(json.loads(line)['choices'][0]['text'])
    except Exception as e: texts.append(f'<parse error: {e}>')
try: texts.append(json.load(open(os.path.join(out,'smoke_chat.json')))['choices'][0]['message']['content'])
except Exception as e: texts.append(f'<parse error: {e}>')
with open(os.path.join(out,'smoke_texts.txt'),'w') as f:
    for i,t in enumerate(texts,1): f.write(f'--- Q{i} ---\n{t}\n\n')
print('\n'.join(t[:90].replace('\n','⏎') for t in texts))
PY
}

# bench <cell_dir> <model> <tokenizer_host_path> <concurrency> <num_prompts>
bench() {
  local out=$1 model=$2 tok=$3 conc=$4 n=$5
  ( nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > "$out/gpu_util.csv" 2>&1 & echo $! > "$out/.gpu_mon" )
  ( bash "$CPU_SAMPLER" "$out/cpu_util.txt" & echo $! > "$out/.cpu_mon" )
  date +%s > "$out/bench_t0.txt"
  docker exec "$BENCH_CN" vllm bench serve --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model "$model" --tokenizer "$tok" \
    --dataset-name sonnet --dataset-path /tmp/sonnet.txt \
    --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 \
    --num-prompts "$n" --max-concurrency "$conc" --request-rate inf --seed 42 \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 \
    > "$out/bench.log" 2>&1
  local rc=$?
  date +%s > "$out/bench_t1.txt"
  kill "$(cat "$out/.gpu_mon")" 2>/dev/null; kill "$(cat "$out/.cpu_mon")" 2>/dev/null
  grep -E "Successful requests|Failed requests|Benchmark duration|Output token throughput|Total token throughput|Median TTFT|P95 TTFT|Median TPOT|P95 TPOT" "$out/bench.log" | sed 's/  */ /g' | tee "$out/bench_summary.txt"
  return $rc
}

cpu_summary() {
  python3 - "$1" <<'PY'
import sys,statistics as st
v=[float(l.split('busy=')[1]) for l in open(sys.argv[1]) if 'busy=' in l]
print(f"CPU busy n={len(v)} avg={sum(v)/len(v):.2f} max={max(v):.2f} p50={st.median(v):.2f}" if v else "CPU no samples")
PY
}
