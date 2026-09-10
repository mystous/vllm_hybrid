#!/usr/bin/env bash
# E11 1단계 종료 후의 단일 순차 대기열 (동시 부팅 충돌 방지).
#  1) IDE_063 NaN 원인 분리 (최우선)
#  2) IDE_061 hot expert 수 H 재탐색
#  3) IDE_062 저부하 토큰지연 판별
#  4) EXP-E11 3단계 12후보 전수측정 (예측 사전등록 커밋 대기)
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
until grep -q E11_S1_DONE $SP/seq_after_e12.log 2>/dev/null; do sleep 60; done
echo "=== 꼬리 대기열 시작 $(date +%H:%M) ==="
echo "=== [1/5] IDE_063 NaN 원인 분리 $(date +%H:%M) ==="; bash $SP/run_ide063.sh
echo "=== [2/5] EXP-E03 보정 (고정 동시성 DRAM 재측정) $(date +%H:%M) ==="; bash $SP/run_e03fix.sh
echo "=== [3/5] IDE_061 H 스윕 $(date +%H:%M) ==="; bash $SP/run_hsweep.sh
echo "=== [4/5] IDE_062 저부하 지연 판별 $(date +%H:%M) ==="; bash $SP/run_ide062.sh
echo "=== [5/5] E11 3단계 대기 (예측 커밋 확인) $(date +%H:%M) ==="
until [ -f $SP/predictions_committed.flag ]; do sleep 60; done
bash $SP/run_e11_stage3.sh
echo "SEQ_TAIL_DONE"
