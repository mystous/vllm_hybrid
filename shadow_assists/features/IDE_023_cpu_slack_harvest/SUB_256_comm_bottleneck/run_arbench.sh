#!/usr/bin/env bash
# SUB_256 iter1: all_reduce 직접 측정 — NCCL Ring/Tree/NVLS 격리 비교 (decode 크기).
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_256_comm_bottleneck
TR=/home/mystous/vllm_dev_prj/bin/torchrun
LOG=$DIR/runs; mkdir -p $LOG
run(){ local tag=$1; shift
  echo "===== $tag ====="
  env "$@" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $TR --nproc_per_node=8 --master_port=29555 \
    $DIR/exp/all_reduce_bench.py 2>$LOG/ar_${tag}.err | grep -aE "#####|size_MB|^[ ]*[0-9]" | tee $LOG/ar_${tag}.txt
}
run "default"
run "ring"  NCCL_ALGO=Ring
run "tree"  NCCL_ALGO=Tree
run "nvls"  NCCL_NVLS_ENABLE=1 NCCL_ALGO=NVLS
echo "===== 완료 ====="
