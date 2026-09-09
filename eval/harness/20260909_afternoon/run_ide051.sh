#!/usr/bin/env bash
# IDE_051: gather+양자화 융합 — 빌드·decode 위상 분해 (base vs FUSE) ·prefill 규모·수치검증·서빙 A/B (운영점 A)
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide051_fuse_qin; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
docker cp $SP/fuse_qin_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/fuse_qin_patch.py | tee -a $B/RUN.log
echo "== build $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_051.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_051.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_051.log' || { echo "BUILD_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_051.log | head -24 | tee -a $B/RUN.log; echo "IDE051_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED' | tee -a $B/RUN.log
echo "BUILD_OK $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker cp $SP/mb480_phase_decode.py sgl-kt:/tmp/; docker cp $SP/mb480_phase.py sgl-kt:/tmp/; docker cp $SP/mb480_numcheck.py sgl-kt:/tmp/
for mode in base fuse base fuse; do env="KT_PHASE_PROF=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_AVX_RB=1 KT_AVX_PF=2"; [ $mode = fuse ] && env="$env KT_FUSE_QIN=1"
  echo "== decode phase ($mode) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_phase_decode.py /tmp/phd_$mode.json 2>&1 | grep -aE '== rows|Profiling Results \(numa\[0\]\)|n_cold=|Traceback'" | tee -a $B/RUN.log; done
for mode in base fuse; do env="KT_PHASE_PROF=1 KT_AMX_MIN_QLEN=0 KT_AVX_RB=1 KT_AVX_PF=2"; [ $mode = fuse ] && env="$env KT_FUSE_QIN=1"
  echo "== prefill phase ($mode) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_phase.py /tmp/php_$mode.json 2>&1 | grep -aE 'numa\[0\].*qlen: 1024|n_cold= 64 T= 1024|Traceback'" | tee -a $B/RUN.log; done
for mode in base fuse; do env="KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_AVX_RB=1 KT_AVX_PF=2"; [ $mode = fuse ] && env="$env KT_FUSE_QIN=1"
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_numcheck.py /tmp/numf_$mode.pt > /tmp/numf_$mode.log 2>&1"; done
docker exec sgl-kt python3 -c "
import torch; b=torch.load('/tmp/numf_base.pt'); o=torch.load('/tmp/numf_fuse.pt')
for k in b: d=(o[k]-b[k]).abs(); print('numcheck fuse T=%d n_cold=%d max_abs_diff=%.3e' % (k[0], k[1], d.max()))" | tee -a $B/RUN.log
echo "IDE051_DONE $B"
