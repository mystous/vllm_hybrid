#!/usr/bin/env python
"""IDE_023 13-lever sweep aggregator.

Reads runs/*.json (per-tag throughput summary) and emits:
  - SUMMARY.md  (in PoC root)  : Δ% vs baseline table + net-positive list
  - results.csv (in PoC root) : tag, output_tps, gpu_util, cpu_util, status
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

POC = Path(__file__).resolve().parent.parent
RUNS = POC / "runs"
SUMMARY = POC / "SUMMARY.md"
CSV_OUT = POC / "results.csv"

LEVERS = ["N1", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11",
          "N14", "N17", "N19", "N20"]
LEVER_DESC = {
    "N1":  "AVX-512 BPE encode",
    "N4":  "SoA paged attention layout",
    "N5":  "SMT-pair pinning scheduler",
    "N6":  "Lock-free priority queue",
    "N7":  "Huge pages 2MB for KV",
    "N8":  "NUMA-local draft state",
    "N9":  "DSA memcpy host<->pinned",
    "N10": "AVX-512 simdjson request parse",
    "N11": "AVX-512 base64 output streaming",
    "N14": "Prefetch suffix tree",
    "N17": "CMT-driven priority (Intel PCM)",
    "N19": "AVX-512 SSE writer",
    "N20": "LogGP admission (cost-aware)",
}


def _apply_status(L: str) -> str:
    """Scrape the per-lever apply line from its boot log."""
    f = POC / "logs" / f"lever_{L}_boot.log"
    if not f.exists():
        return ""
    try:
        txt = f.read_text(errors="ignore")
    except Exception:  # noqa: BLE001
        return ""
    for line in txt.splitlines():
        # Lines look like:  ... [IDE_023] N1: applied (...)
        if f"[IDE_023] {L}: " in line:
            i = line.find(f"[IDE_023] {L}:")
            return line[i + len(f"[IDE_023] {L}: "):].strip()
    return ""


def load(tag: str) -> dict | None:
    f = RUNS / f"{tag}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    baseline = load("baseline")
    base_tps = (baseline or {}).get("output_tps")
    rows: list[dict] = []
    for L in LEVERS:
        tag = f"lever_{L}"
        d = load(tag)
        if d is None:
            row = {"lever": L, "desc": LEVER_DESC[L], "tps": None,
                   "delta_pct": None, "status": "missing",
                   "gpu_util": None, "cpu_util": None}
        elif d.get("status") == "boot_fail":
            row = {"lever": L, "desc": LEVER_DESC[L], "tps": None,
                   "delta_pct": None, "status": "boot_fail",
                   "gpu_util": None, "cpu_util": None}
        else:
            tps = d.get("output_tps")
            delta = None
            if base_tps and tps:
                delta = round((tps / base_tps - 1) * 100, 2)
            row = {"lever": L, "desc": LEVER_DESC[L], "tps": tps,
                   "delta_pct": delta,
                   "status": "ok" if tps else "no_tps",
                   "gpu_util": d.get("gpu_util"),
                   "cpu_util": d.get("cpu_util")}
        rows.append(row)

    # CSV
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lever", "desc", "tps",
                                          "delta_pct", "status",
                                          "gpu_util", "cpu_util"])
        w.writeheader()
        if baseline:
            w.writerow({"lever": "baseline", "desc": "Optimal Config (vanilla+FaP+L2+L10)",
                        "tps": base_tps, "delta_pct": 0.0, "status": "ok",
                        "gpu_util": baseline.get("gpu_util"),
                        "cpu_util": baseline.get("cpu_util")})
        for r in rows:
            w.writerow(r)

    # MD summary
    lines = [
        "# IDE_023 13-Lever PoC — SUB_201",
        "",
        f"- Model: Llama-3.1-8B-Instruct, TP=8, B200 ×8",
        f"- Bench: sharegpt 200p × conc=16 × max-tok=512",
        f"- Baseline = Optimal Config (vanilla + FaP + L2 + L10)",
        "",
        "## Baseline",
        "",
    ]
    if baseline:
        lines += [
            f"- output_tps = **{base_tps}** tps",
            f"- gpu_util = {baseline.get('gpu_util')}%",
            f"- cpu_util = {baseline.get('cpu_util')}%",
            f"- wall_total_s = {baseline.get('wall_total_s')}s",
            f"- n_ok = {baseline.get('n_ok')}/{baseline.get('n')}",
        ]
    else:
        lines += ["- **MISSING — baseline run did not complete**"]
    lines += [
        "",
        "## Lever results",
        "",
        "| lever | description | tps | Δ% vs baseline | status | gpu_util | cpu_util | apply |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for r in rows:
        apply = _apply_status(r["lever"])
        # Trim long apply strings
        if len(apply) > 60:
            apply = apply[:57] + "..."
        lines.append(
            f"| {r['lever']} | {r['desc']} | "
            f"{r['tps'] if r['tps'] is not None else '—'} | "
            f"{(str(r['delta_pct'])+' %') if r['delta_pct'] is not None else '—'} | "
            f"{r['status']} | "
            f"{r['gpu_util'] if r['gpu_util'] is not None else '—'} | "
            f"{r['cpu_util'] if r['cpu_util'] is not None else '—'} | "
            f"{apply} |"
        )

    # Net-positive (Δ% ≥ +3%)
    net_pos = [r for r in rows if r["delta_pct"] is not None and r["delta_pct"] >= 3.0]
    na_run = [r for r in rows if r["status"] not in ("ok",)]
    # Levers whose apply line says "na:" — environmental N/A, even if boot OK
    env_na = []
    for r in rows:
        apply = _apply_status(r["lever"])
        if apply.startswith("na:"):
            env_na.append((r["lever"], r["desc"], apply))
    na = na_run
    lines += [
        "",
        "## Net-positive (Δ% ≥ +3%, noise floor)",
        "",
    ]
    if net_pos:
        for r in net_pos:
            lines.append(f"- **{r['lever']}** ({r['desc']}): Δ = +{r['delta_pct']}%")
    else:
        lines.append("- (none above +3% threshold)")
    lines += [
        "",
        "## N/A or missing levers",
        "",
    ]
    if na:
        for r in na:
            lines.append(f"- {r['lever']} ({r['desc']}): {r['status']}")
    else:
        lines.append("- (all 13 produced an output_tps measurement)")
    lines += [
        "",
        "## Environmental N/A (apply step reported `na:`)",
        "",
    ]
    if env_na:
        for L, desc, why in env_na:
            lines.append(f"- **{L}** ({desc}): {why}")
    else:
        lines.append("- (no lever reported environmental N/A in its apply step)")
    lines += [
        "",
        "## Production-ready recommendation (top 3-5)",
        "",
        "Ranking by Δ% (positive only):",
        "",
    ]
    ranked = sorted(
        [r for r in rows if r["delta_pct"] is not None and r["delta_pct"] > 0],
        key=lambda r: r["delta_pct"], reverse=True,
    )
    if ranked:
        for i, r in enumerate(ranked[:5], 1):
            lines.append(f"{i}. **{r['lever']}** ({r['desc']}): Δ = +{r['delta_pct']}%")
    else:
        lines.append("- (no positive Δ%)")

    lines += [
        "",
        "## Artefacts",
        "",
        f"- per-tag throughput summary: `runs/baseline.json`, `runs/lever_N*.json`",
        f"- boot logs: `logs/*_boot.log`",
        f"- bench logs: `logs/*_bench.log`",
        f"- patch: `vllm/v1/spec_decode/ide023_levers.py` + `vllm/envs.py` (13 env flags)",
        f"- harness: `scripts/sweep.sh`",
        "",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY}")
    print(f"wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
