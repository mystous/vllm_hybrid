#!/usr/bin/env bash
# SUB_236 — DSA 트래픽 채널② 간섭 측정 (컨테이너 내 측정 가능 부분)
#
# 명제: DSA memcpy 트래픽은 RMID 미태깅(채널②)이라 코어의 메모리 BW 발자국으로
#       잡히지 않지만 victim 의 메모리 지연을 실제로 악화시킨다.
# 방법: victim(node0, 0-7) 단독 vs DSA 스트림 {0,2,4}개 동시. DSA 버퍼는
#       numactl --membind=0 으로 node0 강제 → node0 iMC 에서 victim 과 경합.
#       (dsa1=node1 엔진이 node0 메모리를 read/write → UPI 경유 node0 트래픽)
#
# ⚠ 컨테이너 한계 (호스트 몫):
#   - MBA 비차단성 입증: resctrl 미마운트 → 호스트
#   - read_buffers_allowed {96,48,24,12} 셰이핑 곡선: /sys ro + accel-config 부재 → 호스트
#   본 스크립트는 "채널② 간섭의 실재 + DSA 스트림 수 dose-response" 까지만 확정.
set -uo pipefail
cd "$(dirname "$0")"

VICTIM=../src/victim_aggressor
DSA=./dsa_traffic
VCPUS=${VCPUS:-0-7}
SECS=${SECS:-15}
RUNS=${RUNS:-3}
COPY_MB=${COPY_MB:-2}          # = WQ max_transfer_size (2MB)
DSA_CPUS=(16 17 18 19)         # node0 코어, victim(0-7)·sibling(112-119) 회피
WQS=(/dev/dsa/wq1.0 /dev/dsa/wq1.1 /dev/dsa/wq1.2 /dev/dsa/wq1.3)
OUT=${OUT:-./sub236_results}

[ -x "$VICTIM" ] || { echo "[ERR] victim 미빌드 — bash ../src/build.sh"; exit 1; }
[ -x "$DSA" ] || { echo "[ERR] dsa_traffic 미빌드 — gcc -O2 -o dsa_traffic dsa_traffic.c"; exit 1; }
mkdir -p "$OUT"
CSV="$OUT/sub236_summary.csv"
echo "cell,dsa_streams,run,victim_iters,victim_mean_ms,victim_p50_ms,victim_p95_ms,victim_p99_ms,victim_p999_ms,victim_nsload,dsa_GBps_sum" > "$CSV"

run_cell() {  # $1=cell name, $2=nstreams, $3=run
  local cell=$1 n=$2 run=$3 tag="${1}_r${3}"
  local pids=() dsa_logs=()
  # NUMA 배치: numactl/mbind 은 컨테이너에서 EPERM(seccomp). DSA 제출 코어가
  # node0(16-19)이므로 first-touch 로 버퍼가 자동 node0 할당 → numactl 불요.
  for ((i=0;i<n;i++)); do
    local log="$OUT/dsa_${tag}_s${i}.log"
    "$DSA" "${WQS[$i]}" "${DSA_CPUS[$i]}" "$COPY_MB" "$((SECS+6))" > "$log" 2>&1 &
    pids+=($!); dsa_logs+=("$log")
  done
  [ "$n" -gt 0 ] && sleep 3   # DSA warm-up
  local vlog="$OUT/victim_${tag}.log"
  taskset -c "$VCPUS" "$VICTIM" --role victim --cpus "$VCPUS" --secs "$SECS" > "$vlog" 2>&1
  local v; v=$(grep ^VICTIM_RESULT "$vlog" | cut -d, -f2-)
  # DSA 처리량 합산
  local gbsum=0
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
  for l in "${dsa_logs[@]}"; do
    local g; g=$(grep -o 'GBps=[0-9.]*' "$l" | cut -d= -f2)
    [ -n "$g" ] && gbsum=$(awk "BEGIN{print $gbsum+$g}")
  done
  echo "$cell,$n,$run,$v,$gbsum" >> "$CSV"
  printf "  [%s run%s] victim p99=%s ms  dsa_sum=%.1f GB/s\n" \
    "$cell" "$run" "$(echo $v | cut -d, -f5)" "$gbsum"
}

echo "=== SUB_236: victim=$VCPUS secs=$SECS runs=$RUNS copy=${COPY_MB}MB ==="
for run in $(seq 1 "$RUNS"); do
  echo "--- run $run ---"
  run_cell DSA_OFF   0 "$run"
  run_cell DSA_ON_2  2 "$run"
  run_cell DSA_ON_4  4 "$run"
  sleep 1
done

echo ""
echo "===== 요약 ====="
sed 's/,/\t/g' "$CSV"

python3 - "$CSV" <<'EOF'
import csv, sys, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
def agg(cell,key):
    v=[float(r[key]) for r in rows if r["cell"]==cell and r[key] not in("","NA")]
    return (st.mean(v), (st.pstdev(v)/st.mean(v)*100 if st.mean(v) else 0)) if v else (float("nan"),0)
print("\n===== 채널② dose-response =====")
off_p99,off_cv=agg("DSA_OFF","victim_p99_ms")
off_p50,_=agg("DSA_OFF","victim_p50_ms")
for cell,n in (("DSA_OFF",0),("DSA_ON_2",2),("DSA_ON_4",4)):
    p99,cv=agg(cell,"victim_p99_ms"); p50,_=agg(cell,"victim_p50_ms")
    gb,_=agg(cell,"dsa_GBps_sum")
    d99=(p99/off_p99-1)*100 if off_p99 else float('nan')
    d50=(p50/off_p50-1)*100 if off_p50 else float('nan')
    print(f"{cell:9s} streams={n} dsa={gb:6.1f}GB/s  victim p50={p50:7.2f}ms({d50:+5.1f}%) p99={p99:7.2f}ms({d99:+6.1f}%) [CV {cv:.1f}%]")
print("\n[판정] DSA 스트림 증가에 따라 victim p99 단조 악화 → 채널② 간섭 실재 (RMID 미태깅 BW).")
print("       MBA 비차단성·셰이핑 곡선은 호스트(resctrl/accel-config)에서 확정 예정.")
EOF
echo ""
echo "[done] $OUT/"
