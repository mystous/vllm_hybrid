#!/usr/bin/env bash
# IDE_050: rb 커널 B 스트림 소프트웨어 prefetch — 빌드·1~4행 스윕 (rb / rb+PF 2 / 4 / 8) + 수치검증. IDE_049b 종료 후.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE049B_DONE" $SP/ide049b.log 2>/dev/null; do sleep 20; done
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide050_avx_prefetch; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
docker cp $SP/avx_pf_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/avx_pf_patch.py | tee -a $B/RUN.log
echo "== build $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_050.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_050.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_050.log' || { echo "BUILD_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_050.log | head -24 | tee -a $B/RUN.log; echo "IDE050_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED' | tee -a $B/RUN.log
echo "BUILD_OK $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker cp $SP/mb480_rows_pf.py sgl-kt:/tmp/; docker cp $SP/mb480_numcheck.py sgl-kt:/tmp/
for mode in rb pf2 pf4 pf8 rb pf2 pf4 pf8; do env="KT_AMX_MIN_QLEN=1000000 KT_AVX_RB=1"; case $mode in pf2) env="$env KT_AVX_PF=2";; pf4) env="$env KT_AVX_PF=4";; pf8) env="$env KT_AVX_PF=8";; esac
  echo "== rows bench ($mode: $env) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_rows_pf.py /tmp/mbp_$mode.json > /tmp/mbp_$mode.log 2>&1"
  docker exec sgl-kt grep -E "n_cold=|Traceback|Error" /tmp/mbp_$mode.log | tee -a $B/RUN.log; done
docker exec sgl-kt bash -c "cd /tmp && KT_AMX_MIN_QLEN=1000000 KT_AVX_RB=1 KT_AVX_PF=4 CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_numcheck.py /tmp/nump_pf4.pt > /tmp/nump_pf4.log 2>&1"
docker exec sgl-kt python3 -c "
import torch; b=torch.load('/tmp/numv_base.pt'); o=torch.load('/tmp/nump_pf4.pt')
for k in b: d=(o[k]-b[k]).abs(); print('numcheck pf4 T=%d n_cold=%d max_abs_diff=%.3e' % (k[0], k[1], d.max()))" | tee -a $B/RUN.log
echo "IDE050_DONE $B"
