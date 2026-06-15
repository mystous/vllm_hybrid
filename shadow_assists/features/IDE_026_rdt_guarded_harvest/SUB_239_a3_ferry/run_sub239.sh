#!/usr/bin/env bash
# SUB_239 FERRY — DSA-운반 NUMA 파이프라인 vs CPU 직접 원격접근
#
# 워크로드: latency-bound offset pointer-chase (prefetch 무력 → 진짜 NUMA 지연 노출).
#   REMOTE: worker(node0)가 node1 src 를 직접 chase
#   FERRY : DSA(wq1.0)로 node1 src → node0 stage 운반 후 node0-로컬 chase
# NUMA 배치는 first-touch (mbind EPERM 회피): touch_cpu 56 = node1.
#
# 지표: ns_per_step(=CPU-busy 지연), e2e_s(운반+연산). FERRY 게이트:
#   (1) ns_per_step(ferry) < ns_per_step(remote)  — CPU 점유 지연 감소
#   (2) e2e_s(ferry) <= e2e_s(remote)             — 운반 오버헤드 상쇄 여부
set -uo pipefail
cd "$(dirname "$0")"
FERRY=./ferry
WS=${WS:-128}
ITERS=${ITERS:-5}
RUNS=${RUNS:-3}
WCPU=${WCPU:-8}            # worker: node0
TOUCH1=${TOUCH1:-56}      # node1 코어로 src first-touch
WQ=${WQ:-/dev/dsa/wq1.0}
OUT=${OUT:-./sub239_results}
[ -x "$FERRY" ] || { echo "[ERR] ferry 미빌드"; exit 1; }
mkdir -p "$OUT"
CSV="$OUT/sub239_summary.csv"
echo "mode,run,src_probe_ns,ns_per_step,cpu_busy_s,ferry_s,e2e_s" > "$CSV"

getf(){ echo "$2" | tr ',' '\n' | grep "^$1=" | cut -d= -f2; }
emit(){ # $1=mode $2=run $3=raw
  local m=$1 r=$2 line=$3
  echo "$m,$r,$(getf src_probe_ns "$line"),$(getf ns_per_step "$line"),$(getf cpu_busy_sum_s "$line"),$(getf ferry_s "$line"),$(getf e2e_s "$line")" >> "$CSV"
}

echo "=== SUB_239 FERRY: ws=${WS}MB iters=$ITERS runs=$RUNS worker=cpu$WCPU src@node1(touch$TOUCH1) ==="
for run in $(seq 1 "$RUNS"); do
  r=$(FERRY_CHASE=1 "$FERRY" remote "$WCPU" "$WS" "$ITERS" "$WQ" "$TOUCH1" 2>&1 | grep ^FERRY_RESULT)
  echo "  [remote run$run] $(echo "$r" | tr ',' '\n' | grep -E 'ns_per_step|e2e_s' | tr '\n' ' ')"
  emit remote "$run" "$r"
  r=$(FERRY_CHASE=1 "$FERRY" ferry  "$WCPU" "$WS" "$ITERS" "$WQ" "$TOUCH1" 2>&1 | grep ^FERRY_RESULT)
  echo "  [ferry  run$run] $(echo "$r" | tr ',' '\n' | grep -E 'ns_per_step|ferry_s|e2e_s' | tr '\n' ' ')"
  emit ferry "$run" "$r"
done

echo ""; echo "===== 요약 ====="; sed 's/,/\t/g' "$CSV"
python3 - "$CSV" <<'EOF'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
def agg(m,k):
    v=[float(r[k]) for r in rows if r["mode"]==m and r[k] not in("","NA")]
    return (st.mean(v),(st.pstdev(v)/st.mean(v)*100 if st.mean(v) else 0)) if v else (float('nan'),0)
rn,rcv=agg("remote","ns_per_step"); fn,fcv=agg("ferry","ns_per_step")
re,_=agg("remote","e2e_s"); fe,_=agg("ferry","e2e_s"); ff,_=agg("ferry","ferry_s")
print("\n===== FERRY 게이트 =====")
print(f"ns_per_step  remote={rn:.2f} (CV{rcv:.1f}%)  ferry={fn:.2f} (CV{fcv:.1f}%)  Δ={(fn/rn-1)*100:+.1f}%")
print(f"  → (1) CPU-busy 지연: ferry 가 remote 대비 {(1-fn/rn)*100:.1f}% 감소  {'[PASS]' if fn<rn else '[FAIL]'}")
print(f"e2e_s        remote={re:.4f}  ferry={fe:.4f} (그중 운반 {ff:.4f})  Δ={(fe/re-1)*100:+.1f}%")
print(f"  → (2) e2e: ferry {'<= remote [PASS]' if fe<=re else '> remote [운반 오버헤드 미상쇄]'}")
EOF
echo ""; echo "[done] $OUT/"
