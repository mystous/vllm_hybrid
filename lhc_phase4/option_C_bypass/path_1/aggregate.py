#!/usr/bin/env python3
"""LHC Phase 4 Option C — Path 1 aggregate.

Walks ``runs/p1_*_bench.json``, computes per-config mean ± std throughput
(requests/s and tokens/s), paired Δ%, and dumps to ``results.csv`` +
``RESULTS.md`` summary. Also extracts prefix-cache hit rate from
``*_bench.log`` to confirm the gate (Δhit ≤ 1pp)."""

from __future__ import annotations
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def _hit_rate_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    # vllm prints something like "Prefix cache hit rate: 12.3%" in bench
    # serve summary OR scheduler logs cumulative `gpu_prefix_cache_hit_rate`.
    rx = re.compile(r"prefix.cache.hit.rate[^0-9]*([0-9.]+)\s*%", re.I)
    for line in log_path.read_text(errors="ignore").splitlines():
        m = rx.search(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def main():
    rows = []
    for bj in sorted(RUNS.glob("p1_*_bench.json")):
        tag = bj.stem.replace("_bench", "")
        # tag format: p1_<workload>_<config>_s<N>
        parts = tag.split("_")
        # Allow workload names with underscores (e.g. chat_prefix).
        # Locate sweep marker.
        try:
            sweep_idx = next(
                i for i, p in enumerate(parts) if re.fullmatch(r"s\d+", p)
            )
        except StopIteration:
            continue
        sweep = int(parts[sweep_idx][1:])
        # Configs may contain underscores: take everything between workload
        # and sweep marker. Workload is parts[1] OR parts[1:k] joined where
        # k = index of first known config token.
        known_configs = ("vanilla", "lhc_amx_c3_prefix")
        cfg_start = None
        for k in range(2, sweep_idx):
            cand = "_".join(parts[k:sweep_idx])
            if cand in known_configs:
                cfg_start = k
                break
        if cfg_start is None:
            continue
        workload = "_".join(parts[1:cfg_start])
        config = "_".join(parts[cfg_start:sweep_idx])

        with bj.open() as f:
            data = json.load(f)
        rps = data.get("request_throughput", float("nan"))
        out_tps = data.get("output_throughput", float("nan"))
        tot_tps = data.get("total_token_throughput", float("nan"))
        hit = _hit_rate_from_log(RUNS / f"{tag}_bench.log")
        rows.append({
            "tag": tag,
            "workload": workload,
            "config": config,
            "sweep": sweep,
            "req_tps": rps,
            "out_tps": out_tps,
            "tot_tps": tot_tps,
            "prefix_hit_pct": hit,
        })

    # CSV.
    csv_path = ROOT / "results.csv"
    with csv_path.open("w") as f:
        f.write("tag,workload,config,sweep,req_tps,out_tps,tot_tps,prefix_hit_pct\n")
        for r in rows:
            hit = r["prefix_hit_pct"]
            hit_str = "" if hit is None else f"{hit:.2f}"
            f.write(
                f"{r['tag']},{r['workload']},{r['config']},{r['sweep']},"
                f"{r['req_tps']:.4f},{r['out_tps']:.4f},{r['tot_tps']:.4f},"
                f"{hit_str}\n"
            )
    print(f"wrote {csv_path}")

    # Summary.
    md = ["# Path 1 — AMX C3 prefix hash chain results\n"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["workload"], r["config"]), []).append(r)

    for workload in sorted({k[0] for k in groups}):
        md.append(f"## Workload: {workload}\n")
        md.append("| config | sweeps | req_tps mean±std | out_tps mean±std | tot_tps mean±std |")
        md.append("|---|---|---|---|---|")
        means = {}
        for cfg in ("vanilla", "lhc_amx_c3_prefix"):
            rs = groups.get((workload, cfg), [])
            if not rs:
                continue
            req = [r["req_tps"] for r in rs]
            out = [r["out_tps"] for r in rs]
            tot = [r["tot_tps"] for r in rs]
            def mstd(xs):
                m = statistics.mean(xs)
                s = statistics.stdev(xs) if len(xs) > 1 else 0.0
                return m, s
            mr, sr = mstd(req); mo, so = mstd(out); mt, st = mstd(tot)
            means[cfg] = (mr, mo, mt)
            md.append(
                f"| {cfg} | {len(rs)} | "
                f"{mr:.2f}±{sr:.2f} | {mo:.2f}±{so:.2f} | {mt:.2f}±{st:.2f} |"
            )
        if "vanilla" in means and "lhc_amx_c3_prefix" in means:
            v = means["vanilla"]; l = means["lhc_amx_c3_prefix"]
            d_req = (l[0]-v[0])/v[0]*100
            d_out = (l[1]-v[1])/v[1]*100
            d_tot = (l[2]-v[2])/v[2]*100
            md.append("")
            md.append(
                f"**Δ (LHC vs vanilla)**: req {d_req:+.2f}%, "
                f"out {d_out:+.2f}%, tot {d_tot:+.2f}%"
            )
        md.append("")
    md_path = ROOT / "RESULTS.md"
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
