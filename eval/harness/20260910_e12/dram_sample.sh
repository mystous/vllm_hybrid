#!/usr/bin/env bash
# 물리 DRAM 대역폭 샘플러 (소켓별 read/write GB/s). perf uncore_imc cas_count 사용.
# usage: dram_sample.sh <윈도우초> <반복횟수> <출력파일>
W=${1:-10}; N=${2:-1}; OUT=${3:-/dev/stdout}
EV=$(for i in 0 1 2 3 4 5 6 7; do printf "uncore_imc_%s/cas_count_read/,uncore_imc_%s/cas_count_write/," $i $i; done | sed 's/,$//')
for ((k=0;k<N;k++)); do
  sudo perf stat -a --per-socket -x, -e "$EV" -- sleep $W 2>&1 | python3 -c '
import sys, collections
agg=collections.defaultdict(float)
for line in sys.stdin:
    f=line.strip().split(",")
    if len(f)<6: continue
    sock=f[0].strip()
    try: val=float(f[2].replace(",", "")) * 1048576.0   # perf 는 MiB 로 낸다
    except ValueError: continue
    ev=[x for x in f if "cas_count" in x]
    if not ev: continue
    kind="read" if "read" in ev[0] else "write"
    agg[(sock,kind)]+=val
import datetime, os
w=float(os.environ.get("W","10"))
ts=datetime.datetime.now().strftime("%H:%M:%S")
socks=sorted({s for s,_ in agg})
parts=[]
tot=0.0
for s in socks:
    r=agg[(s,"read")]/1e9/w; wr=agg[(s,"write")]/1e9/w
    tot+=r+wr
    parts.append(f"{s} r={r:6.1f} w={wr:5.1f} GB/s")
print(f"{ts} " + " | ".join(parts) + f" | 합 {tot:6.1f} GB/s")
' W=$W
done >> $OUT 2>&1
