"""L7 — Simulation bench: oracle-routed multi-instance vs default single-method.

핵심 가정 (TSK_042 oracle-fact):
  - 각 (model_family, workload_type) 에 대해 4 method 의 measured `output_tps` 가 있다.
  - throughput 은 cell-local (한 시점에 한 instance 가 한 corpus 만 처리).
  - 단일 cluster 에 충분한 GPU 가 있어 모든 family 가 동시에 serve 된다 가정.

비교 scheme:
  - "default": 클러스터 전체가 **vanilla** (no spec, no router) — 가장 흔한 baseline.
  - "best_uniform": 클러스터 전체를 한 method 로 강제. 평균 가장 좋은 method 채택.
                   (예: suffix 가 sharegpt 에서 best 라고 모두 suffix 로) ← static SD config.
  - "oracle":   각 cell 마다 best_method dispatch (L7).

총 throughput 추정:
  C_total = Σ_{(family, workload)} fraction(family, workload) × tps(family, workload, scheme)

`fraction` 은 두 시나리오:
  A) uniform   — 모든 (family × workload) 쌍이 동일 weight.
  B) realistic — 모델 mix 와 corpus mix 가 production traffic 분포 (편향).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from oracle_table import METHODS, MODEL_FAMILY, WORKLOAD_TYPES, Oracle
from router import OracleRouter

OUTDIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Workload mix definitions
# ---------------------------------------------------------------------------
def uniform_workload() -> dict[tuple[str, str], float]:
    """All (family, workload) 동일 weight."""
    families = list(MODEL_FAMILY.values())
    cells = [(f, w) for f in families for w in WORKLOAD_TYPES]
    w = 1.0 / len(cells)
    return {c: w for c in cells}


def realistic_workload() -> dict[tuple[str, str], float]:
    """Production-flavor mix.

    근거 (관측치 + 일반 통념):
      - chat / 대화 (Qwen-7B/32B, Llama-8B, DS-Qwen-7B): 60%
      - long-context lmsys/wildchat (32B/70B): 20%
      - code (humaneval/mbpp/swebench): 15%
      - heavy reasoning (R1 671B / Llama-405B mix): 5%
    """
    family_share = {
        "Qwen-7B": 0.22,
        "Qwen-32B": 0.10,
        "Qwen-72B": 0.04,
        "Llama-8B": 0.18,
        "Llama-70B": 0.07,
        "Llama-405B": 0.02,
        "DS-Qwen-7B": 0.12,
        "DS-Qwen-32B": 0.10,
        "DS-Llama-70B": 0.10,
        "DS-R1-671B": 0.05,
    }
    workload_share = {
        "sharegpt": 0.22,
        "lmsys": 0.18,
        "wildchat": 0.14,
        "humaneval": 0.10,
        "mbpp": 0.08,
        "swebench": 0.08,
        "mix": 0.20,
    }
    out = {}
    for f, fw in family_share.items():
        for w, ww in workload_share.items():
            out[(f, w)] = fw * ww
    # 정규화 안전장치
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


# ---------------------------------------------------------------------------
# Scheme cluster TPS computation
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    scheme: str
    workload_label: str
    cluster_tps_weighted: float       # Σ fraction × tps   (mean tps, weight-weighted)
    cluster_tps_capacity: float       # Σ tps              (raw capacity sum)
    per_cell_tps: dict                # debug
    coverage: int                     # cells contributing (non-NaN)


def cluster_tps(
    oracle: Oracle,
    workload_mix: dict[tuple[str, str], float],
    scheme: str,
    force_method: str | None = None,
) -> ClusterResult:
    """scheme:
      - 'default'      = 모든 cell 에서 vanilla
      - 'static_best'  = workload_mix 가중평균이 가장 큰 단일 method 채택
      - 'oracle'       = cell 마다 best_method dispatch
    """
    per_cell: dict[tuple[str, str], float] = {}
    coverage = 0
    sum_weighted = 0.0
    sum_capacity = 0.0

    rows = {(r.model_family, r.workload_type): r for r in oracle.iter_rows()}
    for (mf, wt), frac in workload_mix.items():
        r = rows[(mf, wt)]
        if scheme == "default":
            tps = r.by_method.get("vanilla", float("nan"))
        elif scheme == "static_best":
            tps = r.by_method.get(force_method, float("nan"))
        elif scheme == "oracle":
            tps = r.best_tps
        else:
            raise ValueError(scheme)
        per_cell[(mf, wt)] = tps
        if tps == tps:  # not NaN
            sum_weighted += frac * tps
            sum_capacity += tps
            coverage += 1

    return ClusterResult(
        scheme=scheme + (f"({force_method})" if force_method else ""),
        workload_label="",
        cluster_tps_weighted=sum_weighted,
        cluster_tps_capacity=sum_capacity,
        per_cell_tps=per_cell,
        coverage=coverage,
    )


def find_best_static_method(
    oracle: Oracle, workload_mix: dict[tuple[str, str], float]
) -> tuple[str, float]:
    best, best_tps = None, -1.0
    for m in METHODS:
        cr = cluster_tps(oracle, workload_mix, "static_best", force_method=m)
        if cr.cluster_tps_weighted > best_tps:
            best, best_tps = m, cr.cluster_tps_weighted
    return best, best_tps


# ---------------------------------------------------------------------------
# Dispatch latency simulation (CPU overhead audit)
# ---------------------------------------------------------------------------
def simulate_dispatch_latency(
    oracle: Oracle,
    workload_mix: dict[tuple[str, str], float],
    n_requests: int = 1_000_000,
    seed: int = 0,
) -> dict:
    router = OracleRouter(oracle)
    rng = random.Random(seed)
    cells = list(workload_mix.keys())
    weights = list(workload_mix.values())

    # warmup
    for _ in range(10_000):
        c = rng.choices(cells, weights=weights, k=1)[0]
        router.dispatch(*c)

    t0 = time.perf_counter_ns()
    for _ in range(n_requests):
        c = rng.choices(cells, weights=weights, k=1)[0]
        router.dispatch(*c)
    t1 = time.perf_counter_ns()
    total_s = (t1 - t0) / 1e9
    return {
        "n_requests": n_requests,
        "total_s": total_s,
        "ns_per_dispatch": (t1 - t0) / n_requests,
        "qps_max_single_core": n_requests / total_s,
        "lookup_ns_internal_mean": router.lookup_latency_ns_mean(),
        "dispatch_counts_sample": dict(
            list(router.dispatch_counts.most_common(8))
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(outdir: Path):
    oracle = Oracle.from_csv()

    workloads = {
        "uniform": uniform_workload(),
        "realistic": realistic_workload(),
    }

    summary_rows = []
    full = {}
    for wl_name, mix in workloads.items():
        # default (vanilla everywhere)
        cr_default = cluster_tps(oracle, mix, "default")
        cr_default.workload_label = wl_name

        # static-best (cluster-wide single method, 가중평균 최대)
        best_m, best_static_tps = find_best_static_method(oracle, mix)
        cr_static = cluster_tps(oracle, mix, "static_best", force_method=best_m)
        cr_static.workload_label = wl_name

        # oracle (cell 마다 best)
        cr_oracle = cluster_tps(oracle, mix, "oracle")
        cr_oracle.workload_label = wl_name

        # 라우터 latency
        lat = simulate_dispatch_latency(oracle, mix, n_requests=1_000_000)

        delta_vs_default = (
            (cr_oracle.cluster_tps_weighted - cr_default.cluster_tps_weighted)
            / cr_default.cluster_tps_weighted * 100.0
        )
        delta_vs_static_best = (
            (cr_oracle.cluster_tps_weighted - cr_static.cluster_tps_weighted)
            / cr_static.cluster_tps_weighted * 100.0
        )

        summary_rows.append(
            {
                "workload": wl_name,
                "default_tps_weighted": cr_default.cluster_tps_weighted,
                "static_best_method": best_m,
                "static_best_tps_weighted": cr_static.cluster_tps_weighted,
                "oracle_tps_weighted": cr_oracle.cluster_tps_weighted,
                "delta_vs_default_pct": delta_vs_default,
                "delta_vs_static_best_pct": delta_vs_static_best,
                "ns_per_dispatch": lat["ns_per_dispatch"],
                "qps_max_single_core": lat["qps_max_single_core"],
            }
        )

        full[wl_name] = {
            "default": {
                "tps_weighted": cr_default.cluster_tps_weighted,
                "tps_capacity": cr_default.cluster_tps_capacity,
                "coverage": cr_default.coverage,
            },
            "static_best": {
                "method": best_m,
                "tps_weighted": cr_static.cluster_tps_weighted,
                "tps_capacity": cr_static.cluster_tps_capacity,
                "coverage": cr_static.coverage,
            },
            "oracle": {
                "tps_weighted": cr_oracle.cluster_tps_weighted,
                "tps_capacity": cr_oracle.cluster_tps_capacity,
                "coverage": cr_oracle.coverage,
            },
            "router_latency": lat,
            "delta_oracle_vs_default_pct": delta_vs_default,
            "delta_oracle_vs_static_best_pct": delta_vs_static_best,
        }

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))
    out_csv = outdir / "bench_summary.csv"
    df.to_csv(out_csv, index=False)
    out_json = outdir / "bench_full.json"
    out_json.write_text(json.dumps(full, indent=2, sort_keys=True))
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_json}")

    # oracle table snapshot
    oracle_csv = outdir / "oracle_table.csv"
    oracle.to_dataframe().to_csv(oracle_csv, index=False)
    print(f"wrote {oracle_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=OUTDIR)
    args = p.parse_args()
    run(args.outdir)
