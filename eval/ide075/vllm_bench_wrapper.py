#!/usr/bin/env python3
"""IDE_075 §7.1 — vllm bench serve 를 그대로 쓰되 (같은 요청 코드·같은 부하), benchmark() 진입 시점에 CLIENT_READY/START barrier 와
perf_counter↔CLOCK_REALTIME anchor 를 추가하는 래퍼 (컨테이너 vllm-h100). 저장 결과의 per-request start_times(perf_counter)·ttfts·itls 를 anchor 로 realtime ns 로 변환 가능.
사용: IDE075_CTL=<ctl_prefix> python3 vllm_bench_wrapper.py <vllm bench serve 인자들...>
동작: 데이터셋·토크나이저 준비 뒤 benchmark() 진입 → <ctl>.ready 생성 → <ctl>.start 대기 → anchor 기록(<ctl>.anchor.json) → 원래 benchmark() 실행 (vllm 의 test prompt 1건 포함)."""
import os, sys, time, json, asyncio, argparse
import vllm.benchmarks.serve as S
from vllm.utils.argparse_utils import FlexibleArgumentParser

CTL = os.environ["IDE075_CTL"]
_orig = S.benchmark


async def _patched(*a, **kw):
    open(CTL + ".ready", "w").write(str(time.time_ns()))
    while not os.path.exists(CTL + ".start"): await asyncio.sleep(0.005)
    anchor = {"realtime_ns": time.time_ns(), "perf_counter": time.perf_counter(), "monotonic_ns": time.monotonic_ns(), "note": "benchmark() 진입 직후; start_times(perf_counter) → realtime_ns + (st − perf_counter)*1e9"}
    json.dump(anchor, open(CTL + ".anchor.json", "w"))
    return await _orig(*a, **kw)


S.benchmark = _patched
parser = FlexibleArgumentParser(description="vllm bench serve (IDE_075 barrier wrapper)")
S.add_cli_args(parser)
args = parser.parse_args(sys.argv[1:])
S.main(args)
