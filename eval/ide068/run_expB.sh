#!/usr/bin/env bash
# IDE_068 / TSK_050 — 실험 B: Qwen3-Coder-480B-A35B-FP8 를 오프로딩으로 GPU 몇 장까지 줄일 수 있는가.
# 셀: b0 GPU-only TP8+EP8 기준선 / b0a GPU-only TP4 (OOM 재실증) / b1 하이브리드 TP4 / b2 TP2 / b3 TP1
# 각 셀 = 부팅 → greedy 4문항 → sonnet 512/128 C16 64req 벤치 + CPU·GPU 샘플러 + DRAM·HBM 사용량.
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide068_expB_480b
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --attention-backend triton --trust-remote-code"
KT="--disable-cuda-graph --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 0"
log "== IDE_068 expB start $TS model=$M_CN =="
log "turbo: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo) max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
stop_server

run_cell() {   # run_cell <name> <gpus> <timeout> <conc> <n> <args...>
  local name=$1 gpus=$2 tmo=$3 conc=$4 n=$5; shift 5
  local out=$BASE/$name
  log "-- cell $name --"
  boot "$out" "$gpus" "$tmo" "$@"
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    log "smoke:"; smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    log "bench C$conc n=$n:"; bench "$out" "$MODEL" "$TOK" "$conc" "$n" | tee -a "$RUN_LOG"
    cpu_summary "$out/cpu_util.txt" | tee -a "$RUN_LOG"
    docker exec "$CN" bash -c "tail -150 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  fi
  stop_server
}

# b0 — GPU-only TP8 + EP8 (순수 TP8 은 FP8 block 제약으로 sglang 이 분할 불가 → EP 로 우회)
run_cell b0_gpu_tp8_ep8 0,1,2,3,4,5,6,7 2400 16 64 $COMMON --tp 8 --ep-size 8 --mem-fraction-static 0.90 --max-total-tokens 131072

# b0a — GPU-only TP4 (TSK_047 OOM 재실증)
run_cell b0a_gpu_tp4 0,1,2,3 900 16 64 $COMMON --tp 4 --mem-fraction-static 0.92

# b1 — 하이브리드 TP4 (TSK_047 재현: 44.5 tok/s)
run_cell b1_hyb_tp4 0,1,2,3 3600 16 64 $COMMON --tp 4 --mem-fraction-static 0.80 --max-total-tokens 131072 $KT

# b2 — 하이브리드 TP2
run_cell b2_hyb_tp2 0,1 3600 16 64 $COMMON --tp 2 --mem-fraction-static 0.80 --max-total-tokens 65536 $KT

# b3 — 하이브리드 TP1
run_cell b3_hyb_tp1 0 3600 16 64 $COMMON --tp 1 --mem-fraction-static 0.80 --max-total-tokens 32768 $KT

log "== expB done $(date +%H:%M:%S) =="
echo "$BASE"
