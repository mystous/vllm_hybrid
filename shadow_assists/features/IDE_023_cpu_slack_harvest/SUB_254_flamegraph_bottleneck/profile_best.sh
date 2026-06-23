#!/usr/bin/env bash
# SUB_254: best-performance(NVFP4 awqgptq + suffix spec + FaP + uniform K-pad) TP8 서빙을
# py-spy 플레임그래프로 프로파일 → 병목 지점. SR-001(spec+fap+pad)+SR-003/004(FP4) 스택.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_254_flamegraph_bottleneck
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
PYSPY=/home/mystous/vllm_dev_prj/bin/py-spy
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8030; LOGD=$DIR/runs
SLOG=$LOGD/serve.log
mkdir -p $LOGD
echo "[boot] best config TP8 (suffix spec K6 + FaP + uniform pad) on $MODEL"
VLLM_SUFFIX_PAD_UNIFORM=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache setsid $VBIN serve $MODEL \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"suffix","num_speculative_tokens":6}' \
  --port $PORT > $SLOG 2>&1 &
L=$!; ok=0
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }
  grep -qiE "Traceback \(most|AssertionError|RuntimeError|unrecognized arg|ValueError|out of memory|Error" $SLOG && { grep -iE "error|assert|traceback" $SLOG|tail -3; }
  sleep 5
done
if [ $ok -ne 1 ]; then echo "[FAIL] 부팅 실패"; tail -20 $SLOG; kill -KILL -- -$L 2>/dev/null; exit 1; fi
echo "[ready] 서버 가동. warmup + 지속 부하 시작"
# warmup
$PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
# 지속 부하 (프로파일 창 덮도록 큰 reqs) 백그라운드
$PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 512 --reqs 600 --tag load --salt p > $LOGD/bench.log 2>&1 &
BENCH=$!
sleep 8  # 정상상태 decode 진입 대기
# 가장 바쁜 python 워커 PID 탐색 (vLLM TP 워커)
WPID=$(ps -eo pid,pcpu,comm,args --sort=-pcpu | grep -iE "VLLM|EngineCore|pt_main|python" | grep -v "grep\|py-spy\|bench_unique" | head -1 | awk '{print $1}')
echo "[profile] busy worker pid=$WPID 30초 record"
ps -p $WPID -o pid,pcpu,args | tail -1 | cut -c1-100
# py-spy: flamegraph SVG + speedscope (gil/idle 포함 보려면 --idle)
sudo -n env "PATH=$PATH" $PYSPY record -p $WPID -d 30 -r 100 --idle -o $LOGD/flame_best.svg --format flamegraph 2>$LOGD/pyspy.log || echo "py-spy SVG 실패: $(tail -2 $LOGD/pyspy.log)"
sudo -n env "PATH=$PATH" $PYSPY record -p $WPID -d 20 -r 100 --idle -o $LOGD/flame_best.speedscope.json --format speedscope 2>>$LOGD/pyspy.log || true
# 부하/서버 정리 (정밀 PID)
kill -TERM $BENCH 2>/dev/null
echo "[done] flamegraph 저장: $LOGD/flame_best.svg / .speedscope.json"
echo "=== bench 결과 ==="; grep -oE "gen_tps=[0-9.]+" $LOGD/bench.log | tail -1
kill -KILL -- -$L 2>/dev/null
echo "[cleanup] 서버 종료"
