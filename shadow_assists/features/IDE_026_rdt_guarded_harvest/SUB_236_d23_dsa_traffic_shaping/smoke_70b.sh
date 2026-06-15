#!/usr/bin/env bash
# SUB_236 스모크 — 70B 실서빙 victim 에 DSA harvest 간섭 + read_buffers shaping.
# 셀: (A) 서빙 단독 tps / (B) +DSA harvest(4스트림) / (C) +DSA harvest + read_buffers=24 shaping.
set -uo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-8021}; MODEL=${MODEL:-meta-llama/Llama-3.1-70B-Instruct}
PY=/home/mystous/vllm_dev_prj/bin/python
LOAD=../SUB_239_a3_ferry/ferry_e2e_load.py
CONC=${CONC:-24}; PTOK=${PTOK:-2000}; MTOK=${MTOK:-128}; REQS=${REQS:-48}
DSA_CPUS=(16 17 18 19); WQS=(/dev/dsa/wq1.0 /dev/dsa/wq1.1 /dev/dsa/wq1.2 /dev/dsa/wq1.3)
OUT=host_smoke; mkdir -p $OUT

run_load(){  # $1=tag
  $PY $LOAD --base http://127.0.0.1:$PORT --model $MODEL \
    --concurrency $CONC --prompt-tokens $PTOK --max-tokens $MTOK --requests $REQS \
    --tag "$1" --out $OUT/smoke236.txt 2>&1 | grep -E "FERRY_E2E|throughput"
}
start_dsa(){ for i in 0 1 2 3; do sudo taskset -c ${DSA_CPUS[$i]} ./dsa_traffic ${WQS[$i]} ${DSA_CPUS[$i]} 2 200 >$OUT/dsa_$i.log 2>&1 & done; sleep 3; }
stop_dsa(){ sudo pkill -x dsa_traffic 2>/dev/null; sleep 2; }
set_rb(){ # $1 = read_buffers_allowed
  sudo accel-config disable-wq dsa1/wq1.0 dsa1/wq1.1 dsa1/wq1.2 dsa1/wq1.3 >/dev/null 2>&1
  sudo accel-config disable-device dsa1 >/dev/null 2>&1
  sudo accel-config config-group dsa1/group1.0 --read-buffers-allowed=$1 >/dev/null 2>&1
  sudo accel-config enable-device dsa1 >/dev/null 2>&1
  for i in 0 1 2 3; do sudo accel-config enable-wq dsa1/wq1.$i >/dev/null 2>&1; done
}

echo "=== SUB_236 스모크 (70B victim, conc=$CONC ptok=$PTOK mtok=$MTOK reqs=$REQS) ==="
echo "[A] 서빙 단독"; run_load A
echo "[B] +DSA harvest (read_buffers=96 기본)"; start_dsa; run_load B; stop_dsa
echo "[C] +DSA harvest + read_buffers=24 shaping"; set_rb 24; start_dsa; run_load C; stop_dsa; set_rb 96
echo "=== 요약 ==="; grep FERRY_E2E $OUT/smoke236.txt | sed -E 's/.*tag=([ABC]),.*gen_tps=([0-9.]+).*/  cell \1: gen_tps=\2/'
