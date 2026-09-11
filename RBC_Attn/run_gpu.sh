#!/usr/bin/env bash
# GPU 벤치 일괄 (지시서 §6).
set -uo pipefail
cd "$(dirname "$0")"
bash build.sh
python3 -m pytest -q tests/test_host.py tests/test_native_metadata.py
python3 -m pytest -q tests/test_gpu.py
bash run_matrix.sh
python3 stream_bench.py --routing-source gpu --nq 2048 --chunk-q 256 --out results/pipeline.json
