#!/usr/bin/env python3
"""SUB_201 L2 — baseline vs prefetch_on 결과 정리 + Δ 계산."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

KEYS = [
    ("total_token_throughput", "Total tok/s"),
    ("output_throughput", "Output tok/s"),
    ("request_throughput", "Req/s"),
    ("max_output_tokens_per_s", "Peak out tok/s"),
    ("mean_ttft_ms", "TTFT mean (ms)"),
    ("median_ttft_ms", "TTFT p50 (ms)"),
    ("p99_ttft_ms", "TTFT p99 (ms)"),
    ("mean_tpot_ms", "TPOT mean (ms)"),
    ("median_tpot_ms", "TPOT p50 (ms)"),
    ("p99_tpot_ms", "TPOT p99 (ms)"),
    ("median_itl_ms", "ITL p50 (ms)"),
    ("median_e2el_ms", "E2EL p50 (ms)"),
    ("p99_e2el_ms", "E2EL p99 (ms)"),
    ("duration", "duration (s)"),
]

HIGHER_BETTER = {
    "total_token_throughput",
    "output_throughput",
    "request_throughput",
    "max_output_tokens_per_s",
}


def load(name: str) -> dict:
    p = RUNS / f"bench_{name}.json"
    if not p.exists():
        sys.exit(f"missing: {p}")
    return json.loads(p.read_text())


def fmt_delta(base: float, on: float, higher_better: bool) -> str:
    if base == 0:
        return "n/a"
    delta_pct = (on - base) / base * 100.0
    sign = "+" if delta_pct >= 0 else ""
    arrow = ""
    if higher_better:
        arrow = " ↑" if delta_pct > 0 else (" ↓" if delta_pct < 0 else "")
    else:
        arrow = " ↑" if delta_pct < 0 else (" ↓" if delta_pct > 0 else "")
    return f"{sign}{delta_pct:.2f}%{arrow}"


def main() -> None:
    base = load("baseline")
    on = load("prefetch_on")
    print("| metric | baseline | prefetch_on | Δ% |")
    print("|---|---:|---:|---:|")
    for key, label in KEYS:
        bv = base.get(key)
        ov = on.get(key)
        if bv is None or ov is None:
            continue
        delta = fmt_delta(float(bv), float(ov), key in HIGHER_BETTER)
        print(f"| {label} | {bv:.2f} | {ov:.2f} | {delta} |")


if __name__ == "__main__":
    main()
