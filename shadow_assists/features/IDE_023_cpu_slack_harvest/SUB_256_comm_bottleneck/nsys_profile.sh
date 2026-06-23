#!/usr/bin/env bash
# SUB_256 iter5-prep: nsys로 best 구성 decode GPU 타임라인 프로파일 → AR 커널 GPU-시간 비중 + 버블.
# custom PTX 커널 빌드 가치(천장) 판정. 통신 직접 측정.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_256_comm_bottleneck
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
NSYS=/usr/local/cuda/bin/nsys
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8036; LOGD=$DIR/runs
mkdir -p $LOGD
# 1) 부하 워처: 서버 ready 되면 nsys 캡처창 동안 지속 부하
( for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 5; done
  echo "[load] ready, 지속 부하 시작 $(date -u +%H:%M:%S)"
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 64 --tag W --salt w >/dev/null 2>&1
  for r in $(seq 1 6); do $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 512 --reqs 400 --tag L$r --salt p$r > $LOGD/nload_$r.txt 2>&1; done
) &
LOADER=$!
# 2) nsys가 serve 래핑, boot(~180s) 후 steady 창 캡처 (--delay 후 --duration)
echo "[nsys] serve 래핑 시작 $(date -u +%H:%M:%S)"
VLLM_SUFFIX_PAD_UNIFORM=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache \
  $NSYS profile --trace=cuda --delay=210 --duration=18 --sample=none --cpuctxsw=none \
    --output=$LOGD/decode_prof --force-overwrite=true \
    $VBIN serve $MODEL --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":6}' \
    --port $PORT > $LOGD/nsys_serve.log 2>&1 &
NPID=$!
# 캡처(--duration=18) 종료 후 nsys가 .nsys-rep 를 쓴다 — 파일이 나타나고 "크기 안정"될 때까지 대기 (serve 살려둠)
prev=-1
for i in $(seq 1 90); do
  if [ -f $LOGD/decode_prof.nsys-rep ]; then
    sz=$(stat -c %s $LOGD/decode_prof.nsys-rep 2>/dev/null||echo 0)
    [ "$sz" = "$prev" ] && [ "$sz" -gt 1000 ] && { echo "[nsys] rep 안정 ($sz bytes)"; break; }
    prev=$sz
  fi
  sleep 5
done
sleep 3
# 이제 정리 (정밀)
for p in $(pgrep -f "awqgptq.*8036|nsys-launcher|nsys-tee"); do cmd=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null); case "$cmd" in *docker*) :;; *) kill -TERM $p 2>/dev/null;; esac; done
kill $LOADER 2>/dev/null; sleep 8
# 3) GPU 커널 요약 → AR vs GEMM 비중
echo "===== GPU 커널 요약 ====="
$NSYS stats --report cuda_gpu_kern_sum --format table $LOGD/decode_prof.nsys-rep 2>/dev/null | head -45
echo "[done]"
