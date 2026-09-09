#!/usr/bin/env bash
# IDE_046 수치 검증 (rows 스윕 뒤): base vs bcache vs bcache+aloadd 출력 비교
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE046_DONE" $SP/ide046.log 2>/dev/null; do sleep 20; done
grep -q "BUILD_FAIL" $SP/ide046.log && { echo "skip (build fail)"; echo "IDE046N_DONE"; exit 0; }
B=$(ls -d $REPO/eval/results/*_ide046_amx_bcache | tail -1)
docker cp $SP/mb480_numcheck.py sgl-kt:/tmp/
for mode in base bcache bcache_aloadd; do env="KT_AMX_MIN_QLEN=0"; [ $mode = bcache ] && env="$env KT_AMX_BCACHE=1"; [ $mode = bcache_aloadd ] && env="$env KT_AMX_BCACHE=1 KT_AMX_A_LOADD=1"
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_numcheck.py /tmp/num_$mode.pt > /tmp/num_$mode.log 2>&1"; docker exec sgl-kt tail -2 /tmp/num_$mode.log | tee -a $B/RUN.log; done
docker exec sgl-kt python3 -c "
import torch
b=torch.load('/tmp/num_base.pt')
for m in ('bcache','bcache_aloadd'):
    o=torch.load('/tmp/num_%s.pt'%m)
    for k in b: d=(o[k]-b[k]).abs(); print('numcheck %s T=%d n_cold=%d max_abs_diff=%.3e mean|base|=%.3e rel=%.2e' % (m, k[0], k[1], d.max(), b[k].abs().mean(), d.max()/b[k].abs().max()))
" | tee -a $B/RUN.log
echo "IDE046N_DONE"
