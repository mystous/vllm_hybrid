#!/usr/bin/env python3
"""Extract per-workload prefix-cache hit rate from vllm /metrics dumps.

Reads each *_metrics_before.txt and *_metrics_after.txt under a results
directory and computes (hits_after - hits_before) / (queries_after - queries_before).

Usage:
    extract_prefix_hit_rates.py <runs_dir>
"""
from __future__ import annotations
import re, sys
from pathlib import Path


def parse_metrics(path: Path) -> dict[str, float]:
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        # vllm:prefix_cache_queries_total{...} 53793.0
        m = re.match(r"(vllm:[^{]+)\{[^}]*\}\s+(\S+)", line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        try:
            out[name] = float(val)
        except ValueError:
            continue
    return out


def hit_rate(before: dict, after: dict) -> float | None:
    Qa = after.get("vllm:prefix_cache_queries_total")
    Qb = before.get("vllm:prefix_cache_queries_total")
    Ha = after.get("vllm:prefix_cache_hits_total")
    Hb = before.get("vllm:prefix_cache_hits_total")
    if None in (Qa, Qb, Ha, Hb):
        return None
    dQ = Qa - Qb
    dH = Ha - Hb
    if dQ <= 0:
        return None
    return dH / dQ


def main():
    if len(sys.argv) != 2:
        print("usage: extract_prefix_hit_rates.py <runs_dir>", file=sys.stderr)
        sys.exit(2)
    d = Path(sys.argv[1])
    print(f"{'tag':<48} {'queries Δ':>12} {'hits Δ':>10} {'hit rate':>10}")
    print("-" * 84)
    for before in sorted(d.glob("*_metrics_before.txt")):
        tag = before.name.replace("_metrics_before.txt", "")
        after = d / f"{tag}_metrics_after.txt"
        b = parse_metrics(before)
        a = parse_metrics(after)
        if not b or not a:
            continue
        Qa = a.get("vllm:prefix_cache_queries_total", 0)
        Qb = b.get("vllm:prefix_cache_queries_total", 0)
        Ha = a.get("vllm:prefix_cache_hits_total", 0)
        Hb = b.get("vllm:prefix_cache_hits_total", 0)
        dQ = Qa - Qb
        dH = Ha - Hb
        hr = (dH / dQ) if dQ > 0 else float("nan")
        print(f"{tag:<48} {dQ:>12.0f} {dH:>10.0f} {hr:>10.4f}")


if __name__ == "__main__":
    main()
