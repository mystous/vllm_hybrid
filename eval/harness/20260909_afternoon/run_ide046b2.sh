#!/usr/bin/env bash
# IDE_046-b: (1) HT 실험: 현재 빌드로 KT_THREADS 96 vs 192 (2풀) rows 128 phase; (2) q_input/q_down 병렬화 패치 → 빌드 → phase (base / QA_PAR / QA_PAR+BCACHE)
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
B=$(ls -d $REPO/eval/results/*_ide046_amx_bcache | tail -1)
docker cp $SP/mb480_phase.py sgl-kt:/tmp/
echo "== build2 $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_046b.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_046b.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_046b.log' || { echo "BUILD_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_046b.log | head -24 | tee -a $B/RUN.log; echo "IDE046B_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED' | tee -a $B/RUN.log
echo "BUILD2_OK $(date +%H:%M:%S)" | tee -a $B/RUN.log
for mode in base qapar qapar_bcache; do env="KT_PHASE_PROF=1 KT_AMX_MIN_QLEN=0"; [ $mode = qapar ] && env="$env KT_QA_PAR=1"; [ $mode = qapar_bcache ] && env="$env KT_QA_PAR=1 KT_AMX_BCACHE=1"
  echo "== phase2 ($mode: $env) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_phase.py /tmp/phase2_$mode.json 2>&1 | grep -E '== rows|Profiling|n_cold=|Traceback|Error'" | tee -a $B/RUN.log; done
# 수치 검증 (qapar+bcache vs base)
docker cp $SP/mb480_numcheck.py sgl-kt:/tmp/
for mode in base qapar_bcache; do env="KT_AMX_MIN_QLEN=0"; [ $mode = qapar_bcache ] && env="$env KT_QA_PAR=1 KT_AMX_BCACHE=1"
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_numcheck.py /tmp/num2_$mode.pt > /tmp/num2_$mode.log 2>&1"; done
docker exec sgl-kt python3 -c "
import torch; b=torch.load('/tmp/num2_base.pt'); o=torch.load('/tmp/num2_qapar_bcache.pt')
for k in b: d=(o[k]-b[k]).abs(); print('numcheck2 qapar_bcache T=%d n_cold=%d max_abs_diff=%.3e rel=%.2e' % (k[0], k[1], d.max(), d.max()/b[k].abs().max()))" | tee -a $B/RUN.log
echo "IDE046B_DONE"
