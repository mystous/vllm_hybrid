#!/usr/bin/env bash
# IDE_046: AMX INT4 B 타일 언팩 캐시 — 패치·빌드·rows 스윕 (base / bcache / bcache+A loadd), 모두 AMX 경로 (KT_AMX_MIN_QLEN=0)
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide046_amx_bcache; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
docker cp $SP/amx_bcache_patch.py sgl-kt:/tmp/ && docker exec sgl-kt python3 /tmp/amx_bcache_patch.py | tee -a $B/RUN.log
echo "== build $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'cd /sgl-workspace/ktransformers/kt-kernel && rm -f /tmp/ktwheel/*.whl && pip wheel . -w /tmp/ktwheel --no-deps --no-build-isolation --no-cache-dir > /tmp/ktbuild_046.log 2>&1; echo BUILD_EXIT=$? >> /tmp/ktbuild_046.log'
docker exec sgl-kt bash -c 'grep -q "BUILD_EXIT=0" /tmp/ktbuild_046.log' || { echo "BUILD_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m3 -B2 -A4 "error:" /tmp/ktbuild_046.log | head -24 | tee -a $B/RUN.log; echo "IDE046_DONE"; exit 1; }
docker exec sgl-kt bash -c 'cd /tmp && rm -rf ktwheel_x && mkdir ktwheel_x && cd ktwheel_x && python3 -m zipfile -e /tmp/ktwheel/kt_kernel-*.whl . && cp kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/kt_kernel/ && echo INSTALLED' | tee -a $B/RUN.log
echo "BUILD_OK $(date +%H:%M:%S)" | tee -a $B/RUN.log
docker cp $SP/mb480_rows3.py sgl-kt:/tmp/
for mode in base bcache bcache_aloadd; do env="KT_AMX_MIN_QLEN=0"; [ $mode = bcache ] && env="$env KT_AMX_BCACHE=1"; [ $mode = bcache_aloadd ] && env="$env KT_AMX_BCACHE=1 KT_AMX_A_LOADD=1"
  echo "== rows bench ($mode: $env) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_rows3.py /tmp/mb3_$mode.json > /tmp/mb3_$mode.log 2>&1"; docker cp sgl-kt:/tmp/mb3_$mode.json $B/ 2>/dev/null
  docker exec sgl-kt grep -E "n_cold=|Traceback|Error" /tmp/mb3_$mode.log | tee -a $B/RUN.log; done
echo "IDE046_DONE $B"
