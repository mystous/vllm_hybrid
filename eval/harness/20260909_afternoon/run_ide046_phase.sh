#!/usr/bin/env bash
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE046N_DONE" $SP/ide046n.log 2>/dev/null; do sleep 15; done
B=$(ls -d $REPO/eval/results/*_ide046_amx_bcache | tail -1)
docker cp $SP/mb480_phase.py sgl-kt:/tmp/
for mode in base bcache; do env="KT_PHASE_PROF=1 KT_AMX_MIN_QLEN=0"; [ $mode = bcache ] && env="$env KT_AMX_BCACHE=1"
  echo "== phase ($mode)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && $env CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_phase.py /tmp/phase_$mode.json 2>&1 | grep -E '== rows|Profiling|n_cold=|Traceback'" | tee -a $B/RUN.log; done
echo "IDE046P_DONE"
