#!/usr/bin/env bash
# IDE_048: AVX-512 vec 경로 레지스터 블로킹 — 패치·빌드·rows 스윕 (vec 강제) base vs KT_AVX_RB=1 + 수치검증. IDE_047d 종료 후.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE047C_DONE" $SP/ide047d.log 2>/dev/null; do sleep 20; done
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide048_avx_rb; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
docker cp $SP/avx_rb_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/avx_rb_patch.py | tee -a $B/RUN.log
echo "== build $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_048.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_048.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_048.log' || { echo "BUILD_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_048.log | head -24 | tee -a $B/RUN.log; echo "IDE048_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED' | tee -a $B/RUN.log
echo "BUILD_OK $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker cp $SP/mb480_rows_vec.py sgl-kt:/tmp/; docker cp $SP/mb480_numcheck.py sgl-kt:/tmp/
for mode in base rb base rb; do env="KT_AMX_MIN_QLEN=1000000"; [ $mode = rb ] && env="$env KT_AVX_RB=1"
  echo "== rows bench vec ($mode: $env) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_rows_vec.py /tmp/mbv_$mode.json > /tmp/mbv_$mode.log 2>&1"; docker cp sgl-kt:/tmp/mbv_$mode.json $B/mbv_${mode}_$(date +%H%M%S).json 2>/dev/null
  docker exec sgl-kt grep -E "n_cold=|Traceback|Error" /tmp/mbv_$mode.log | tee -a $B/RUN.log; done
for mode in base rb; do env="KT_AMX_MIN_QLEN=1000000"; [ $mode = rb ] && env="$env KT_AVX_RB=1"
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_numcheck.py /tmp/numv_$mode.pt > /tmp/numv_$mode.log 2>&1"; docker exec sgl-kt tail -1 /tmp/numv_$mode.log | tee -a $B/RUN.log; done
docker exec sgl-kt python3 -c "
import torch; b=torch.load('/tmp/numv_base.pt'); o=torch.load('/tmp/numv_rb.pt')
for k in b: d=(o[k]-b[k]).abs(); print('numcheck rb T=%d n_cold=%d max_abs_diff=%.3e rel=%.2e' % (k[0], k[1], d.max(), d.max()/b[k].abs().max()))" | tee -a $B/RUN.log
echo "IDE048_DONE $B"
