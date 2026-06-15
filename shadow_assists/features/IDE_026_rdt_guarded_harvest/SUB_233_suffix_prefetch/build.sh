#!/usr/bin/env bash
# SUB_233 suffix tree prefetch — 확장 빌드 (baseline/prefetch 공통 플래그)
set -eu
SRC=$1; OUT=$2
NB=/home/mystous/vllm_dev_prj/lib/python3.12/site-packages/nanobind; PYINC=/usr/include/python3.12
g++ -O3 -DNDEBUG -shared -fPIC -std=c++20 -fvisibility=hidden -fno-strict-aliasing \
  -I"$NB/include" -I"$NB/ext/robin_map/include" -I"$PYINC" -I"$SRC" \
  "$SRC/bindings.cc" "$SRC/suffix_tree.cc" "$NB/src/nb_combined.cpp" \
  -o "$OUT"
