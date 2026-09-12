#!/usr/bin/env bash
# 지시서 §3 비교 행렬. 통과한 작은 구간부터 확장한다.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results
export RBC_THREADS=${RBC_THREADS:-16} OMP_PROC_BIND=close OMP_PLACES=cores
run() {
  local tag=$1; shift
  echo "=== $tag $(date +%H:%M:%S) ==="
  python3 bench.py "$@" --fresh --native required --reps 20 --out "results/$tag.json" 2>&1 | tail -22
}
# (a) query 수 축 — 문맥 64k 고정
for NQ in 256 512 2048; do
  run "m_nq${NQ}" --pattern common_private --nq $NQ --nk 65536 --g 8 --d 128 --sel 16
done
# (b) 문맥 축 — query 512 고정
for NK in 16384 65536 131072; do
  run "m_nk${NK}" --pattern common_private --nq 512 --nk $NK --g 8 --d 128 --sel 16
done
# (c) GQA group 축
for G in 4 8 16; do
  run "m_g${G}" --pattern common_private --nq 512 --nk 65536 --g $G --d 128 --sel 16
done
# (d) 선택 block 수 축
for SEL in 8 16 32; do
  run "m_sel${SEL}" --pattern common_private --nq 512 --nk 65536 --g 8 --d 128 --sel $SEL
done
# (e) 반례 패턴
for P in random clustered disjoint; do
  run "m_pat_${P}" --pattern $P --nq 512 --nk 65536 --g 8 --d 128 --sel 16
done
echo "MATRIX_DONE"
