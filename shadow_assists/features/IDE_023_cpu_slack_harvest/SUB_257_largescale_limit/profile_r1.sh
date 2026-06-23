#!/usr/bin/env bash
# SUB_257 iter1: R1-671B MoE 한계 측정 — nsys GPU 커널 breakdown으로 지배 병목 규명
# (all-to-all expert comm? grouped expert GEMM? attention?). 신규 알고리즘 타깃 결정용.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_257_largescale_limit
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm; NSYS=/usr/local/cuda/bin/nsys
MODEL=deepseek-ai/DeepSeek-R1; PORT=8050; LOGD=$DIR/runs; mkdir -p $LOGD
# 부하 워처: ready되면 nsys 캡처창 동안 지속 부하
( for i in $(seq 1 200); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 6; done
  echo "[load] ready $(date -u +%H:%M:%S)"
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 32 --ptok 1000 --mtok 64 --reqs 64 --tag W --salt w >/dev/null 2>&1
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 32 --ptok 1000 --mtok 512 --reqs 6000 --tag LONG --salt pp > $LOGD/load_long.txt 2>&1
) &
LOADER=$!
echo "[nsys] R1-671B EP8 serve 래핑 $(date -u +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache HF_MODULES_CACHE=/home/mystous/hf_mods \
  $NSYS profile --trace=cuda --delay=300 --duration=25 --sample=none --cpuctxsw=none \
    --output=$LOGD/r1_prof --force-overwrite=true \
    $VBIN serve $MODEL --tensor-parallel-size 8 --enable-expert-parallel \
    --gpu-memory-utilization 0.92 --max-model-len 8192 --enforce-eager \
    --port $PORT > $LOGD/nsys_serve.log 2>&1 &
NPID=$!
# rep 안정될 때까지 대기 (serve 살려둠)
prev=-1
for i in $(seq 1 140); do
  if [ -f $LOGD/r1_prof.nsys-rep ]; then sz=$(stat -c %s $LOGD/r1_prof.nsys-rep 2>/dev/null||echo 0); [ "$sz" = "$prev" ] && [ "$sz" -gt 1000 ] && { echo "[nsys] rep 안정 $sz"; break; }; prev=$sz; fi
  # 부팅 실패 조기 감지
  grep -qiE "Traceback \(most|out of memory|Error.*init|RuntimeError" $LOGD/nsys_serve.log 2>/dev/null && grep -qiE "raise|Error" $LOGD/nsys_serve.log && { tail -5 $LOGD/nsys_serve.log; }
  sleep 8
done
sleep 3
for p in $(pgrep -f "DeepSeek-R1.*8050|nsys-launcher|nsys-tee"); do cmd=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null); case "$cmd" in *docker*) :;; *) kill -TERM $p 2>/dev/null;; esac; done
kill $LOADER 2>/dev/null; sleep 10
echo "===== GPU 커널 요약 (R1-671B) ====="
$NSYS stats --report cuda_gpu_kern_sum --format table $LOGD/r1_prof.nsys-rep 2>/dev/null | head -42
echo "[done]"
