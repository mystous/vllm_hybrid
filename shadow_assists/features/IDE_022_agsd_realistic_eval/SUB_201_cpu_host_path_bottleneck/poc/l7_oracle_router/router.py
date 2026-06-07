"""L7 — CPU-side oracle router for multi-model vLLM serving.

routing scheme (production):
  request → router.dispatch(model_family, workload_type) → instance_url

본 모듈은 두 가지 모드를 제공한다.
  1) in-process Router  — 단위 lookup latency 측정 + bench.py simulation 의 라이브러리.
  2) FastAPI app        — reverse-proxy 미니멈 데모 (실 boot 안 해도 latency 측정 가능).

CPU 점유:
  - lookup 은 dict get → 평균 < 1 µs.
  - workload_type 은 클라이언트가 헤더 (`X-Workload-Type`) 로 보내거나
    server-side 가벼운 regex 로 추정한다. (본 PoC 는 헤더 가정 — TSK_044 와 격리)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from oracle_table import METHODS, MODEL_FAMILY, WORKLOAD_TYPES, Oracle


@dataclass
class DispatchDecision:
    model_family: str
    workload_type: str
    chosen_method: str
    instance_key: str          # f"{model_family}|{chosen_method}"
    expected_tps: float
    default_tps: float
    uplift_pct: float


class OracleRouter:
    """CPU-only O(1) router. multi-model cluster 의 단일 in-memory state."""

    def __init__(self, oracle: Oracle):
        self._oracle = oracle
        # 인스턴스 키 → 누적 dispatch 수
        self.dispatch_counts: Counter[str] = Counter()
        # 의사 결정 시각 누적 — latency micro-bench
        self._lookup_ns_total = 0
        self._lookup_calls = 0

    def dispatch(self, model_family: str, workload_type: str) -> DispatchDecision:
        t0 = time.perf_counter_ns()
        row = self._oracle.lookup(model_family, workload_type)
        t1 = time.perf_counter_ns()
        self._lookup_ns_total += t1 - t0
        self._lookup_calls += 1

        instance_key = f"{model_family}|{row.best_method}"
        self.dispatch_counts[instance_key] += 1
        return DispatchDecision(
            model_family=model_family,
            workload_type=workload_type,
            chosen_method=row.best_method,
            instance_key=instance_key,
            expected_tps=row.best_tps,
            default_tps=row.default_tps,
            uplift_pct=row.uplift_pct,
        )

    # latency micro-bench --------------------------------------------------
    def lookup_latency_ns_mean(self) -> float:
        return self._lookup_ns_total / max(1, self._lookup_calls)

    def lookup_calls(self) -> int:
        return self._lookup_calls


# ---------------------------------------------------------------------------
# CLI: lookup latency micro-bench  (10 M lookups, 무작위 분포)
# ---------------------------------------------------------------------------
def micro_bench(n: int = 10_000_000, seed: int = 0) -> dict:
    import random

    r = random.Random(seed)
    oracle = Oracle.from_csv()
    router = OracleRouter(oracle)
    families = list(MODEL_FAMILY.values())
    workloads = WORKLOAD_TYPES

    # warmup
    for _ in range(10_000):
        router.dispatch(r.choice(families), r.choice(workloads))

    t0 = time.perf_counter_ns()
    for _ in range(n):
        router.dispatch(r.choice(families), r.choice(workloads))
    t1 = time.perf_counter_ns()

    total_s = (t1 - t0) / 1e9
    ns_per_call = (t1 - t0) / n
    qps = n / total_s
    return {
        "n": n,
        "total_s": total_s,
        "ns_per_call": ns_per_call,
        "qps": qps,
        "model_families": len(families),
        "workload_types": len(workloads),
        "lookup_calls_internal": router.lookup_calls(),
        "lookup_ns_mean_internal": router.lookup_latency_ns_mean(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = micro_bench(args.n, args.seed)
    print(json.dumps(out, indent=2))
