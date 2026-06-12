#!/usr/bin/env bash
# IDE_026 / TSK_048 T1 — CPU-only 간섭 재현 + CAT/MBA 격리 A/B (호스트 root 실행)
#
# 셀 정의 (task.md T1 + test.md G1 의 IDLE baseline 추가):
#   IDLE : victim 단독 (aggressor off)            — G1 "간섭 실재" 의 기준선
#   B0   : 공유 (CLOS 분리 없음, 기본 그룹)        — 간섭 노출
#   B1   : CAT 분리 (serving 상위 way / harvest 하위 4 way), MBA off(=100)
#   B2   : B1 + harvest MBA 20%
#
# 게이트 (test.md G1):
#   간섭 실재 : B0 victim p99 >= IDLE p99 × 1.10
#   L2 GO    : B2 victim p99 회복 >= 80% AND B2 aggressor GB/s >= B0 의 70%
#
# 사용: sudo bash run_t1_ab.sh   (환경변수로 조정 가능, 아래 참조)
set -uo pipefail
cd "$(dirname "$0")"

RESCTRL=${RESCTRL:-/sys/fs/resctrl}
VICTIM_CPUS=${VICTIM_CPUS:-0-7}        # socket 0 (node0 = 0-55,112-167)
AGGR_CPUS=${AGGR_CPUS:-16-55}          # socket 0 내부 — MBA 는 socket-로컬만 throttle (함정 #3)
SECS=${SECS:-30}
RUNS=${RUNS:-3}
HARVEST_WAYS=${HARVEST_WAYS:-4}
HARVEST_MBA=${HARVEST_MBA:-20}
OUT=${OUT:-./t1_results}
PY=${PY:-python3}
BIN=./victim_aggressor

[ -x "$BIN" ] || { echo "[ERR] $BIN 없음 — 먼저 bash build.sh"; exit 1; }
[ -d "$RESCTRL/info" ] || { echo "[ERR] $RESCTRL 미마운트 — sudo mount -t resctrl resctrl /sys/fs/resctrl"; exit 1; }
[ "$(id -u)" = 0 ] || { echo "[ERR] root 필요"; exit 1; }

mkdir -p "$OUT"
CSV="$OUT/t1_summary.csv"
echo "cell,run,victim_iters,victim_mean_ms,victim_p50_ms,victim_p95_ms,victim_p99_ms,victim_p999_ms,aggr_GBps" > "$CSV"

rdt() { "$PY" ./rdt_ctl.py --root "$RESCTRL" "$@"; }

setup_groups() {  # $1 = cell
    # 초기화: 그룹 제거 (있다면)
    rdt teardown --group serving --group harvest 2>/dev/null || true
    case "$1" in
        IDLE|B0) ;;  # CLOS 분리 없음 — 전부 기본 그룹
        B1)
            rdt setup --group serving --l3-ways top  --harvest-ways "$HARVEST_WAYS" --mb 100
            rdt setup --group harvest --l3-ways low4 --harvest-ways "$HARVEST_WAYS" --mb 100
            rdt assign --group serving --cpus "$VICTIM_CPUS"
            rdt assign --group harvest --cpus "$AGGR_CPUS"
            ;;
        B2)
            rdt setup --group serving --l3-ways top  --harvest-ways "$HARVEST_WAYS" --mb 100
            rdt setup --group harvest --l3-ways low4 --harvest-ways "$HARVEST_WAYS" --mb "$HARVEST_MBA"
            rdt assign --group serving --cpus "$VICTIM_CPUS"
            rdt assign --group harvest --cpus "$AGGR_CPUS"
            ;;
    esac
}

run_cell() {  # $1 = cell, $2 = run idx
    local cell=$1 run=$2
    local tag="${cell}_r${run}"
    echo "=== [$tag] victim=$VICTIM_CPUS aggr=$AGGR_CPUS secs=$SECS ==="
    setup_groups "$cell"

    local aggr_pid="" aggr_log="$OUT/aggr_$tag.log"
    if [ "$cell" != "IDLE" ]; then
        # aggressor 를 victim 보다 10s 길게 — victim 측정 전구간 부하 보장
        taskset -c "$AGGR_CPUS" "$BIN" --role aggressor --cpus "$AGGR_CPUS" \
            --secs $((SECS + 10)) > "$aggr_log" 2>&1 &
        aggr_pid=$!
        sleep 3  # aggressor warm-up
    fi

    # mon_data 병행 기록 (B1/B2 만 그룹 존재; B0/IDLE 은 root 그룹)
    local mon_groups="root"
    [ "$cell" = B1 ] || [ "$cell" = B2 ] && mon_groups="serving,harvest"
    rdt mon --groups "$mon_groups" --interval 1 --duration "$SECS" \
        --csv "$OUT/mon_$tag.csv" --quiet &
    local mon_pid=$!

    local vlog="$OUT/victim_$tag.log"
    taskset -c "$VICTIM_CPUS" "$BIN" --role victim --cpus "$VICTIM_CPUS" \
        --secs "$SECS" > "$vlog" 2>&1
    cat "$vlog"

    wait "$mon_pid" 2>/dev/null || true
    local aggr_gbps="NA"
    if [ -n "$aggr_pid" ]; then
        wait "$aggr_pid" || true
        cat "$aggr_log"
        aggr_gbps=$(grep ^AGGR_RESULT "$aggr_log" | cut -d, -f4)
    fi
    local v
    v=$(grep ^VICTIM_RESULT "$vlog" | cut -d, -f2-)
    echo "$cell,$run,$v,$aggr_gbps" >> "$CSV"
}

for run in $(seq 1 "$RUNS"); do
    for cell in IDLE B0 B1 B2; do
        run_cell "$cell" "$run"
        sleep 2
    done
done

# 정리
rdt teardown --group serving --group harvest 2>/dev/null || true

echo ""
echo "===== T1 요약 ($CSV) ====="
column -s, -t "$CSV"

# 게이트 판정 (run 별 p99 평균으로)
"$PY" - "$CSV" <<'EOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
def agg(cell, key):
    vals = [float(r[key]) for r in rows if r["cell"] == cell and r[key] != "NA"]
    return (st.mean(vals), (st.stdev(vals)/st.mean(vals)*100 if len(vals) > 1 and st.mean(vals) else 0.0)) if vals else (float("nan"), 0.0)
idle_p99, idle_cv = agg("IDLE", "victim_p99_ms")
b0_p99, b0_cv = agg("B0", "victim_p99_ms")
b1_p99, _ = agg("B1", "victim_p99_ms")
b2_p99, b2_cv = agg("B2", "victim_p99_ms")
b0_gbps, _ = agg("B0", "aggr_GBps")
b2_gbps, _ = agg("B2", "aggr_GBps")
print(f"\n===== G1 게이트 판정 =====")
print(f"IDLE p99={idle_p99:.3f}ms (CV {idle_cv:.1f}%)  B0 p99={b0_p99:.3f}ms (CV {b0_cv:.1f}%)")
print(f"B1 p99={b1_p99:.3f}ms  B2 p99={b2_p99:.3f}ms (CV {b2_cv:.1f}%)")
print(f"B0 aggr={b0_gbps:.1f}GB/s  B2 aggr={b2_gbps:.1f}GB/s")
interference = b0_p99 / idle_p99 - 1 if idle_p99 else float("nan")
print(f"[간섭 실재] B0 vs IDLE p99: {interference*100:+.1f}% (게이트: >= +10%) → {'PASS' if interference >= 0.10 else 'FAIL — aggressor 강도 재설계 (1회 한도)'}")
if b0_p99 > idle_p99:
    recovery = (b0_p99 - b2_p99) / (b0_p99 - idle_p99)
    print(f"[L2 회복] B2 회복률: {recovery*100:.1f}% (GO: >= 80%, kill: < 50%)")
aggr_keep = b2_gbps / b0_gbps if b0_gbps else float("nan")
print(f"[harvest 유지] B2/B0 aggressor: {aggr_keep*100:.1f}% (GO: >= 70%)")
EOF
echo ""
echo "[done] 상세 로그·mon_data: $OUT/"
