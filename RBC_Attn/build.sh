#!/usr/bin/env bash
# RBC-Attn 플래너 빌드. OpenMP 있으면 켜고 없으면 단일 스레드로 빌드한다.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
CXX=${CXX:-g++}
FLAGS="-O3 -march=native -std=c++17 -fPIC -shared -Wall -Wextra"
if echo 'int main(){}' | $CXX -fopenmp -x c++ - -o /dev/null 2>/dev/null; then
  FLAGS="$FLAGS -fopenmp"
  echo "OpenMP 사용"
else
  echo "OpenMP 없음 — 단일 스레드로 빌드"
fi
$CXX $FLAGS src/planner.cpp -o build/librbc_planner.so
echo "빌드 완료: build/librbc_planner.so"
