#!/usr/bin/env bash
set -uo pipefail; export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad; REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_probe_decode_phase_threads; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
docker cp $SP/mb480_phase_decode.py sgl-kt:/tmp/; docker cp $SP/mb480_rows_pf.py sgl-kt:/tmp/
echo "== decode phase split (n_cold 24, rows 1/2, 96 threads, RB+PF2)" | tee -a $B/RUN.log
docker exec sgl-kt bash -c "cd /tmp && KT_THREADS=96 KT_POOLS=2 KT_PHASE_PROF=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_AVX_RB=1 KT_AVX_PF=2 CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_phase_decode.py /tmp/phd.json 2>&1 | grep -aE '== rows|Profiling|n_cold=|Traceback'" | tee -a $B/RUN.log
for thr in 96 112 96 112; do echo "== threads=$thr rows 1/2/4 n_cold 32 (RB+PF2)" | tee -a $B/RUN.log
  docker exec sgl-kt bash -c "cd /tmp && KT_THREADS=$thr KT_POOLS=2 KT_AMX_MIN_QLEN=1000000 KT_AVX_RB=1 KT_AVX_PF=2 CUDA_VISIBLE_DEVICES=0 python3 /tmp/mb480_rows_pf.py /tmp/thr$thr.json 2>&1 | grep -aE 'n_cold=|Traceback'" | tee -a $B/RUN.log; done
echo "PROBE_DONE $B"
