#!/usr/bin/env bash
# 70B suffix canonical 부하 중 host 프로세스 py-spy 프로파일 — serving-트랙 표적 선정용
set -u
cd /home/mystous/vllm_hybrid
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
OUT=shadow_assists/features/IDE_026_rdt_guarded_harvest/profiling
PORT=8011
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_SUFFIX_PAD_UNIFORM=1 setsid taskset -c 0-47,56-103 "$VBIN" serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":6}' \
    > "$OUT/boot.log" 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $PID 2>/dev/null || { log DEAD; exit 1; }
    sleep 5
done
log READY

# 부하 시작 (mix 500)
PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
    --in "$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet" --method profile \
    --model meta-llama/Llama-3.1-70B-Instruct --model-tag Llama-3.1-70B-Instruct \
    --port $PORT --max-tokens 8192 --concurrency 32 --limit 500 --shuffle \
    --out "$OUT/summ_profile_mix.json" --raw /dev/null > "$OUT/bench.log" 2>&1 &
BENCH=$!
sleep 30   # 부하 안정화

# 프로세스 식별: APIServer / EngineCore (자식)
ps --ppid $(pgrep -f 'vllm serve' | head -1) -o pid,comm,args 2>/dev/null | head -5 > "$OUT/proc_tree.txt"
# TP 워커 = EngineCore 의 자식 프로세스들 — CPU 사용 최다 자식 선택
ECORE=$(pgrep -f 'EngineCore' | head -1)
ENG=$(ps --ppid "$ECORE" -o pid,%cpu --no-headers 2>/dev/null | sort -k2 -rn | head -1 | awk '{print $1}')
[ -z "$ENG" ] && ENG=$ECORE
ps --ppid "$ECORE" -o pid,%cpu,comm --no-headers > "$OUT/worker_pad_children.txt" 2>/dev/null
API=$(pgrep -f 'vllm serve' | head -1)
log "profiling: api=$API engine=$ENG"
sudo /home/mystous/vllm_dev_prj/bin/py-spy dump --pid $ENG > "$OUT/worker0_pad_threads.txt" 2>&1
sudo /home/mystous/vllm_dev_prj/bin/py-spy record --pid $ENG --duration 120 --rate 200 --native \
    --format speedscope --output "$OUT/worker0_pad_profile.speedscope.json" > "$OUT/pyspy.log" 2>&1
sudo /home/mystous/vllm_dev_prj/bin/py-spy record --pid $API --duration 60 --rate 100 \
    --format speedscope --output "$OUT/api_profile.speedscope.json" >> "$OUT/pyspy.log" 2>&1
log "profile done — bench 종료 대기"
wait $BENCH 2>/dev/null
pg=$(ps -o pgid= -p $PID | tr -d ' '); kill -9 -"$pg" 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 "$p" 2>/dev/null; done
log ALL_DONE
