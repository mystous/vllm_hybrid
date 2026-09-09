#!/usr/bin/env bash
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE051A_DONE" $SP/ide051a.log 2>/dev/null; do sleep 10; done
B=$(ls -d $REPO/eval/results/*_ide051_fuse_qin | tail -1)
for run in fuse2 base2; do env="KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_AVX_RB=1 KT_AVX_PF=2"; [ $run = fuse2 ] && env="$env KT_FUSE_QIN=1"
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_numcheck.py /tmp/numf_$run.pt > /tmp/numf_$run.log 2>&1"; done
docker exec sgl-kt python3 -c "
import torch
def cmp(a,b,tag):
    A=torch.load('/tmp/numf_%s.pt'%a); Bq=torch.load('/tmp/numf_%s.pt'%b)
    for k in A: d=(Bq[k]-A[k]).abs(); print('numcheck %s T=%d n_cold=%d max_abs_diff=%.3e n_bad=%d' % (tag, k[0], k[1], d.max(), int((d>1e-6).sum())))
cmp('base','base2','base-vs-base2'); cmp('fuse','fuse2','fuse-vs-fuse2'); cmp('base','fuse2','base-vs-fuse2')" | tee -a $B/RUN.log
echo "IDE051D_DONE"
