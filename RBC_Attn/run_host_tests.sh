#!/usr/bin/env bash
# 호스트 테스트 (CUDA 불필요). 지시서 §2.
set -euo pipefail
cd "$(dirname "$0")"
[ -f build/librbc_planner.so ] || bash build.sh
python -m pytest -q tests/test_host.py tests/test_native_metadata.py "$@"
