#!/usr/bin/env bash
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE046C_DONE" $SP/ide046c.log 2>/dev/null; do sleep 20; done
B=$(ls -d $REPO/eval/results/*_ide046_amx_bcache | tail -1)
docker cp $SP/mb480_decode_ht.py sgl-kt:/tmp/
for thr in 96 192 96 192; do echo "== decode HT test threads=$thr (vec 경로 기본, KT_AMX_MIN_ROWS=3) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && KT_THREADS=$thr KT_POOLS=2 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_decode_ht.py /tmp/dec_ht$thr.json 2>&1 | grep -E 'n_cold=|Traceback|Error'" | tee -a $B/RUN.log; done
echo "HTDEC_DONE"
