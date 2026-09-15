#!/usr/bin/env bash
# IDE_068 / TSK_051 — 실험 A: 최대 모델 Kimi-K2-Instruct (1.03 TB FP8) 오프로딩 서빙.
# 단계: (1) 다운로드 완료 확인 → (2) kt quant int4 (CPU 96 threads, GPU 실험과 겹치지 않게 실행)
#       → (3) 하이브리드 TP8 부팅 → greedy 4문항 → C8 벤치 + DRAM/HBM 사용량.
# 사용: run_expA.sh [convert|serve|all]
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
STEP=${1:-all}
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide068_expA_kimi_k2
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
HUB=$HOME/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Instruct
SNAP=$(ls -d $HUB/snapshots/*/ 2>/dev/null | head -1)
M_CN=/models/hub/models--moonshotai--Kimi-K2-Instruct/snapshots/$(basename "${SNAP:-none}")
KT_OUT=/models/kt/kimi-k2-int4
MODEL=kimi
log "== IDE_068 expA ($STEP) start $TS =="
log "turbo: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"

# ---- (1) 다운로드 완료 확인 ----
n_inc=$(ls "$HUB/blobs" 2>/dev/null | grep -c incomplete)
n_st=$(ls "$SNAP" 2>/dev/null | grep -c safetensors)
log "download: incomplete=$n_inc safetensors_in_snapshot=$n_st size=$(du -sh "$HUB" | cut -f1)"
if [ "$n_inc" != 0 ] || [ "$n_st" -lt 61 ]; then log "다운로드 미완료 — 중단"; exit 2; fi

# ---- (2) 변환 ----
if [ "$STEP" = convert ] || [ "$STEP" = all ]; then
  if docker exec "$CN" bash -c "test -f $KT_OUT/config.json" 2>/dev/null; then
    log "변환본 존재 — skip"
  else
    log "-- kt quant int4 시작 (96 threads) --"; free -g | head -2 | tee -a "$RUN_LOG"
    t0=$(date +%s)
    docker exec "$CN" bash -c "kt quant $M_CN -m int4 -i fp8 -o $KT_OUT --cpu-threads 96 --numa-nodes 2 -y > /tmp/kt_quant_kimi.log 2>&1; echo QUANT_EXIT=\$? >> /tmp/kt_quant_kimi.log"
    t1=$(date +%s)
    docker exec "$CN" bash -c 'tail -5 /tmp/kt_quant_kimi.log' | tee -a "$RUN_LOG"
    docker exec "$CN" bash -c 'cat /tmp/kt_quant_kimi.log' > "$BASE/kt_quant.log"
    log "변환 소요 $((t1-t0))초"
    docker exec "$CN" bash -c "grep -q QUANT_EXIT=0 /tmp/kt_quant_kimi.log" || { log "QUANT FAIL"; exit 1; }
    docker exec "$CN" bash -c "du -sh $KT_OUT; ls $KT_OUT | wc -l" | tee -a "$RUN_LOG"
  fi
fi

# ---- (3) 서빙 ----
if [ "$STEP" = serve ] || [ "$STEP" = all ]; then
  stop_server
  COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --attention-backend triton --trust-remote-code"
  KT="--disable-cuda-graph --kt-weight-path $KT_OUT --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 0"
  out=$BASE/a1_hyb_tp8
  log "-- cell a1_hyb_tp8 --"
  boot "$out" 0,1,2,3,4,5,6,7 5400 $COMMON --tp 8 --mem-fraction-static 0.80 --max-total-tokens 65536 $KT
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    log "smoke:"; smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    log "bench C8 n=32:"; bench "$out" "$MODEL" "$SNAP" 8 32 | tee -a "$RUN_LOG"
    cpu_summary "$out/cpu_util.txt" | tee -a "$RUN_LOG"
    free -g > "$out/free_after_bench.txt"
    docker exec "$CN" bash -c "tail -150 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  fi
  stop_server
fi
log "== expA ($STEP) done $(date +%H:%M:%S) =="
echo "$BASE"
