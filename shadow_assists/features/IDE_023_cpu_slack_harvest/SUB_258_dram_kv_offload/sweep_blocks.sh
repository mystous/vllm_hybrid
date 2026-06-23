#!/usr/bin/env bash
# 크로스오버 특성화: GPU KV 블록 수를 바꿔가며 (working-set/cache 비율) A/B 비교.
# working-set ≈ 24 prefix × ~3050 tok ≈ 73000 tok ≈ 4560 blocks(16/blk).
# blocks ≫ 4560 → evict 없음 → offload 무용(오버헤드). blocks ≪ → evict 많음 → DRAM win.
set -e
cd /home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_258_dram_kv_offload
PY=/home/mystous/vllm_dev_prj/bin/python
OUT=runs/sweep_blocks.jsonl; : > $OUT
kill_prev(){ for p in $(cat runs/.pids 2>/dev/null); do kill $p 2>/dev/null||true; done; : > runs/.pids; sleep 6; }
wait_ready(){ local port=$1 log=$2; for i in $(seq 1 60); do curl -s http://127.0.0.1:$port/health>/dev/null 2>&1 && return 0; grep -q "initialization failed" $log && return 1; sleep 5; done; return 1; }

for BLK in 800 1500 3000 4800 7000; do
  kill_prev
  CUDA_VISIBLE_DEVICES=1 nohup bash serve.sh gpuonly $BLK 8001 > runs/sw_A_$BLK.log 2>&1 & echo $! >> runs/.pids
  CUDA_VISIBLE_DEVICES=0 nohup bash serve.sh dram   $BLK 8002 > runs/sw_B_$BLK.log 2>&1 & echo $! >> runs/.pids
  wait_ready 8001 runs/sw_A_$BLK.log || { echo "blk=$BLK A fail"; continue; }
  wait_ready 8002 runs/sw_B_$BLK.log || { echo "blk=$BLK B fail"; continue; }
  # 엔진 PID도 기록(정확 kill 위해)
  grep -ohE "EngineCore pid=[0-9]+" runs/sw_A_$BLK.log runs/sw_B_$BLK.log | grep -oE "[0-9]+" >> runs/.pids
  for s in 8001 8002; do $PY bench_kv.py --port $s --model Qwen/Qwen2.5-7B-Instruct --n-prefix 24 --prefix-tokens 3000 --rounds 4 --max-tokens 32 --concurrency 8 --label warm >/dev/null 2>&1; done
  A=$($PY bench_kv.py --port 8001 --model Qwen/Qwen2.5-7B-Instruct --n-prefix 24 --prefix-tokens 3000 --rounds 4 --max-tokens 32 --concurrency 8 --label gpuonly_blk$BLK --out $OUT)
  B=$($PY bench_kv.py --port 8002 --model Qwen/Qwen2.5-7B-Instruct --n-prefix 24 --prefix-tokens 3000 --rounds 4 --max-tokens 32 --concurrency 8 --label dram_blk$BLK --out $OUT)
  echo "blk=$BLK (cache=$((BLK*16))tok)"; echo "  A: $A"; echo "  B: $B"
done
kill_prev
echo "=== sweep done ==="