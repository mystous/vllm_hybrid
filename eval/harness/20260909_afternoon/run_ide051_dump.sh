#!/usr/bin/env bash
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
true
B=$(ls -d $REPO/eval/results/*_ide051_fuse_qin | tail -1)
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
docker cp $SP/dumpA_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/dumpA_patch.py | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_051d.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_051d.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_051d.log' || { echo "BUILD_FAIL"; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_051d.log | head -20; echo "IDE051DUMP3_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED'
docker exec sgl-kt rm -f /tmp/dumpA_*.bin
for mode in base fuse; do env="KT_DUMP_A=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_AVX_RB=1 KT_AVX_PF=2"; [ $mode = fuse ] && env="$env KT_FUSE_QIN=1"
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_numcheck.py /tmp/numd_$mode.pt > /tmp/numd_$mode.log 2>&1"; done
docker exec sgl-kt python3 -c "
import numpy as np, glob, os
def load(fn):
    b=open(fn,'rb').read(); m,mm,kk=np.frombuffer(b[:12],np.int32); off=12
    a=np.frombuffer(b[off:off+mm*kk],np.int8).copy(); off+=mm*kk
    d=np.frombuffer(b[off:off+4*mm],np.float32).copy(); off+=4*mm
    mp=np.frombuffer(b[off:],np.int32).reshape(-1,2)
    return m,mm,kk,a,d,mp
for fb in sorted(glob.glob('/tmp/dumpA_q*_base.bin')):
    ff=fb.replace('_base.bin','_fuse.bin')
    if not os.path.exists(ff): continue
    m,mm,kk,a,d,mp=load(fb); m2,mm2,kk2,a2,d2,mp2=load(ff)
    print(os.path.basename(fb),'m',m,m2,'max_m',mm,mm2,'d_equal',np.array_equal(d[:m],d2[:m]),'d_maxdiff',float(np.abs(d[:m]-d2[:m]).max()),'a_bytes_diff',int((a!=a2).sum()),'of',a.size,'map_equal',np.array_equal(mp,mp2))
    if not np.array_equal(d[:m],d2[:m]):
        bad=np.nonzero(d[:m]!=d2[:m])[0]; print('  rows with d diff:', bad[:10].tolist(), 'd base/fuse', d[bad[:3]].tolist(), d2[bad[:3]].tolist())
    if (a!=a2).sum():
        idx=np.nonzero(a!=a2)[0]; print('  first a diffs at byte idx', idx[:8].tolist(), 'base', a[idx[:8]].tolist(), 'fuse', a2[idx[:8]].tolist())
" | tee -a $B/RUN.log
echo "IDE051DUMP3_DONE"
