#!/usr/bin/env bash
cd /home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_250_ratedistortion_mixedprec/step4
CKPT=/raid/hf_cache/awq_nvfp4_70b
for i in $(seq 1 200); do
  if grep -qa "saved AWQ-NVFP4 checkpoint" runs/make_awq.log 2>/dev/null && [ -f $CKPT/model.safetensors.index.json ]; then break; fi
  sleep 20
done
sleep 8
if [ -f $CKPT/model.safetensors.index.json ]; then
  echo "[awq 체크포인트 OK] sweep 시작"; GPUS=0,1,2,3 bash awq_sweep.sh
else echo "[awq 미생성]"; fi
