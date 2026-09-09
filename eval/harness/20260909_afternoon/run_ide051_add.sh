#!/usr/bin/env bash
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE051_DONE" $SP/ide051.log 2>/dev/null; do sleep 15; done
B=$(ls -d $REPO/eval/results/*_ide051_fuse_qin | tail -1)
docker cp $SP/mb480_phase_decode2.py sgl-kt:/tmp/
for mode in base fuse base fuse; do env="KT_PHASE_PROF=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_AVX_RB=1 KT_AVX_PF=2"; [ $mode = fuse ] && env="$env KT_FUSE_QIN=1"
  echo "== decode phase qlen>=10 (n_cold 32, rows 3/6 → T 12/24) ($mode) $(date +%H:%M:%S)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_phase_decode2.py /tmp/phd2_$mode.json 2>&1 | grep -aE '== rows|Profiling Results \(numa\[0\]\)|n_cold=|Traceback'" | tee -a $B/RUN.log; done
echo "IDE051A_DONE"
bash $SP/run_ide051_serve.sh
