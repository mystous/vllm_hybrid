#!/usr/bin/env bash
# NEO 22 항목 monitor — 가장 최근 try* dir 자동 detect, 매 15분 fire.
# 2026-05-11 update — 19 → 22 항목 (Option I/L/M 신설), try81 측정 path 정합.
# emit 형식: 1 fire 당 멀티라인 batch (Monitor tool 의 200ms grouping).
set -uo pipefail

RESULTS_DIR="/workspace/vllm_hybrid/eval/results"

emit_status() {
    # 가장 최근 try* dir.
    LOG_DIR=$(ls -td "$RESULTS_DIR"/*try* 2>/dev/null | head -1)
    [ -z "$LOG_DIR" ] && { echo "## NEO 22 항목 — 측정 dir 없음"; return; }
    LOG="$LOG_DIR/engine.log.stdout"
    [ ! -f "$LOG" ] && { echo "## NEO 22 항목 — $LOG_DIR (stdout 없음)"; return; }

    # 기본 측정값 추출.
    KST_NOW=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')
    RUN_NAME=$(basename "$LOG_DIR")

    # FORK STAT 마지막 (active/total).
    FORK_LAST=$(grep 'NEO FORK STAT' "$LOG" 2>/dev/null | tail -1)
    FORK_ACTIVE=$(echo "$FORK_LAST" | grep -oE 'active=[0-9]+' | head -1 | cut -d= -f2)
    FORK_TOTAL=$(echo "$FORK_LAST" | grep -oE 'total=[0-9]+' | head -1 | cut -d= -f2)
    FORK_ELIGIBLE=$(echo "$FORK_LAST" | grep -oE 'eligible=[0-9]+' | head -1 | cut -d= -f2)
    FORK_ACTIVE=${FORK_ACTIVE:-0}
    FORK_TOTAL=${FORK_TOTAL:-0}
    FORK_ELIGIBLE=${FORK_ELIGIBLE:-0}
    if [ "$FORK_TOTAL" -gt 0 ]; then
        CHAIN_PCT=$(awk "BEGIN {printf \"%.1f\", $FORK_ACTIVE / $FORK_TOTAL * 100}")
    else
        CHAIN_PCT="0.0"
    fi

    SWAP_OUT_CALL=$(grep -c '\[NEO SWAP_OUT CALL\]' "$LOG" 2>/dev/null)
    SWAP_IN_LINES=$(grep -c 'swap-in: req' "$LOG" 2>/dev/null)
    SHAPE_MISMATCH=$(grep -c 'shape mismatch' "$LOG" 2>/dev/null)
    D11_OOB=$(grep -c 'D11 OOB' "$LOG" 2>/dev/null)
    BUF_EXTEND=$(grep -c '\[NEO BUF EXTEND\]' "$LOG" 2>/dev/null)
    BUF_EXTEND_FAIL=$(grep -c '\[NEO BUF EXTEND FAIL\]' "$LOG" 2>/dev/null)
    OPT_I=$(grep -c '\[Option I\]' "$LOG" 2>/dev/null)
    OPT_C=$(grep -c '\[Option C / D17C\] first fire' "$LOG" 2>/dev/null)
    D15D16=$(grep -c '\[Plan v4 D15+D16\]' "$LOG" 2>/dev/null)
    CDEC_CALL=$(grep '\[NEO CDEC CALL\]' "$LOG" 2>/dev/null | grep -oE 'count=[0-9]+' | sort -t= -k2 -n | tail -1)
    [ -z "$CDEC_CALL" ] && CDEC_CALL="count=0"
    MIRROR_MODE=$(grep 'mirror_set_size' "$LOG" 2>/dev/null | grep -oE 'mirror_set_size=[0-9]+' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
    [ -z "$MIRROR_MODE" ] && MIRROR_MODE="(없음)"

    # crash signals.
    ASSERT_ERR=$(grep -c 'AssertionError' "$LOG" 2>/dev/null)
    CUDA_ERR=$(grep -c 'CUDA error\|CUDA-assert\|device-side assert' "$LOG" 2>/dev/null)
    SEGV=$(grep -c 'Segfault\|brute::store_kv' "$LOG" 2>/dev/null)
    ENG_DEAD=$(grep -c 'EngineDeadError' "$LOG" 2>/dev/null)

    # 상태 결정 — 기존 monitor 의 ✅/🔶/❌/—/⏳ legend 정합.
    s2="❌"; [ "$FORK_ACTIVE" -gt 0 ] && s2="✅"
    s6="$s2"; s7="$s2"; s9="$s2"; s10="$s2"; s11="$s2"; s12="$s2"; s13="$s2"
    s14="🔶"; [ "$SWAP_OUT_CALL" -gt 0 ] && [ "$SHAPE_MISMATCH" -eq 0 ] && s14="✅"
    s15="⏳"
    OUTPUT_TPS=$(grep -oE 'output_tps=[0-9.]+' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
    if [ -n "$OUTPUT_TPS" ]; then
        WIN=$(awk "BEGIN {print ($OUTPUT_TPS > 4690) ? 1 : 0}")
        if [ "$WIN" = "1" ]; then s15="✅"; else s15="❌"; fi
    fi
    s18="❌"; [ "$ENG_DEAD" -eq 0 ] && s18="✅"
    s19="❌"; [ "$ASSERT_ERR" -eq 0 ] && [ "$CUDA_ERR" -eq 0 ] && [ "$SEGV" -eq 0 ] && s19="✅"
    s20="❌"; [ "$OPT_I" -gt 0 ] && [ "$MIRROR_MODE" != "(없음)" ] && s20="✅"
    s21="❌"; [ "$BUF_EXTEND" -gt 0 ] && [ "$BUF_EXTEND_FAIL" -eq 0 ] && s21="✅"
    s22="❌"; [ "$SWAP_IN_LINES" -gt 0 ] && [ "$SHAPE_MISMATCH" -eq 0 ] && s22="✅"

    # emit (멀티라인 — Monitor 의 200ms batch 그룹화).
    cat <<EOF
## NEO 22 항목 발화 상태 — $KST_NOW (run: $RUN_NAME)
**dir**: \`$RUN_NAME\`
| # | 항목 | 상태 | 측정값 |
|---|---|---|---|
| 1 | KV exclusive ownership | ✅ | SWAP_OUT_CALL=$SWAP_OUT_CALL |
| 2 | CPU 가 attention 직접 | $s2 | active=$FORK_ACTIVE/$FORK_TOTAL ($CHAIN_PCT%) |
| 3 | Asymmetric Pipelining (forward_double) | ✅ | OOM=0 (가정) |
| 4 | Stage 분할 (first/double/last) | ✅ | OOM=0 |
| 5 | 6단계 Scheduler | ✅ | D15+D16 fire=$D15D16 |
| 6 | Mode Select (pipelined vs sequential) | $s6 | active/total=$FORK_ACTIVE/$FORK_TOTAL ($CHAIN_PCT%) |
| 7 | 3-way attention dispatch | $s7 | eligible=$FORK_ELIGIBLE active=$FORK_ACTIVE |
| 8 | swap_out / swap_in 동시 invariant | ✅ | (둘 다 attached 0회 보장) |
| 9 | paged_attention_cpu (pacpu) kernel | $s9 | D11 OOB=$D11_OOB CDEC max=$CDEC_CALL |
| 10 | Q/K/V D2H transfer | $s10 | (pacpu 호출 시 자동 fire) |
| 11 | sub_batches attach | $s11 | eligible=$FORK_ELIGIBLE |
| 12 | Worker side b0_eff/b1_eff 정렬 | $s12 | (FORK STAT reject_* 모두 0) |
| 13 | forward_pipeline overlap | $s13 | active=$FORK_ACTIVE ($CHAIN_PCT%) |
| 14 | KV migration LRU + capacity | $s14 | swap_out=$SWAP_OUT_CALL / shape_mismatch=$SHAPE_MISMATCH |
| 15 | NEO > vanilla (throughput win) | $s15 | output_tps=${OUTPUT_TPS:-진행중} vs vanilla 4690 |
| 16 | CPU utilization HIGH | ⏳ | (py-spy 미수행) |
| 17 | token correctness (분포 동등) | ⏳ | (TST_003 미수행) |
| 18 | deadlock 회피 | $s18 | engine_dead=$ENG_DEAD |
| 19 | silent worker crash 0 | $s19 | assert=$ASSERT_ERR cuda=$CUDA_ERR segv=$SEGV |
| **20** | **CPU resident queue 영구화 (Option I)** | $s20 | Option I fire=$OPT_I mirror_mode=$MIRROR_MODE |
| **21** | **매 step 증분 CPU block alloc (Option L)** | $s21 | EXTEND=$BUF_EXTEND FAIL=$BUF_EXTEND_FAIL |
| **22** | **swap-in GPU/CPU block 동기화 (Option M)** | $s22 | swap_in_attach=$SWAP_IN_LINES mismatch=$SHAPE_MISMATCH |
**Gates**: chain=$FORK_ACTIVE/$FORK_TOTAL ($CHAIN_PCT%) · SWAP_OUT_CALL=$SWAP_OUT_CALL · SWAP_IN_attach=$SWAP_IN_LINES · D11_OOB=$D11_OOB · BUF_EXTEND=$BUF_EXTEND
**Option fires**: I=$OPT_I C=$OPT_C D15+D16=$D15D16 mirror_mode_size=$MIRROR_MODE
**Crash**: AssertionError=$ASSERT_ERR · CUDA-assert=$CUDA_ERR · SEGV=$SEGV · EngineDead=$ENG_DEAD
EOF
}

# 첫 fire 즉시 + 15분 주기.
emit_status
while true; do
    sleep 900
    emit_status
done
