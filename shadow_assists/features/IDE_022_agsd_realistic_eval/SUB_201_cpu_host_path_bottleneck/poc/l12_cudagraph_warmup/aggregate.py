"""Aggregate L12 sweep results into a single summary table.

Reads ``runs/V0_*.json`` and ``runs/V1_*.json`` produced by ``burst_bench.py``
and prints a per-phase + per-mode mean/std/min/max TTFT table plus the
predictor accuracy/cost extracted from ``runs/V1_*_predictor.jsonl.rank0``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def _load_runs(runs_dir: str, prefix: str, min_ok: int = 195) -> list[dict]:
    """Load result JSONs, skipping runs that hit serving errors mid-bench.

    A run with n_ok < min_ok is considered failed (engine death, OOM, etc)
    and excluded from aggregation.
    """
    out = []
    for p in sorted(glob.glob(os.path.join(runs_dir, f"{prefix}_*.json"))):
        # skip *_predictor.jsonl etc
        if p.endswith(".json"):
            with open(p) as f:
                d = json.load(f)
            if d.get("n_ok", 0) < min_ok:
                d["_excluded_reason"] = f"n_ok={d.get('n_ok')} < {min_ok}"
                print(
                    f"[aggregate] EXCLUDE {os.path.basename(p)}: "
                    f"{d['_excluded_reason']}"
                )
                continue
            d["_path"] = p
            d["_tag"] = os.path.basename(p)[:-5]
            out.append(d)
    return out


def _summarise(runs: list[dict]) -> dict:
    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tps = []
    for r in runs:
        tps.append(r.get("output_tps", 0))
        for ph, st in r.get("phase_stats", {}).items():
            if st.get("n", 0) > 0:
                for k in ("ttft_p50", "ttft_p90", "ttft_p99", "ttft_max", "ttft_mean"):
                    if st.get(k) is not None:
                        rows[ph][k].append(st[k])
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for ph, mp in rows.items():
        out[ph] = {}
        for k, v in mp.items():
            if not v:
                out[ph][k] = None
                continue
            out[ph][k] = {
                "mean": round(statistics.mean(v), 1),
                "std": round(statistics.stdev(v), 1) if len(v) > 1 else 0.0,
                "min": round(min(v), 1),
                "max": round(max(v), 1),
                "n": len(v),
            }
    out["_tps_mean"] = round(statistics.mean(tps), 1) if tps else 0.0
    out["_tps_std"] = round(statistics.stdev(tps), 1) if len(tps) > 1 else 0.0
    return out


def _load_predictor(runs_dir: str, prefix: str = "V1") -> dict:
    """Aggregate predictor stats across all V1_*_predictor.jsonl.rank0 files."""
    final_records = []
    overhead_obs_ns = []
    overhead_pred_ns = []
    for p in sorted(glob.glob(
        os.path.join(runs_dir, f"{prefix}_*_predictor.jsonl.rank0")
    )):
        if not os.path.exists(p):
            continue
        # take the last record (highest n_steps) as the run's final snapshot
        last = None
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                last = rec
                overhead_obs_ns.append(rec.get("obs_ns_per_step", 0))
                overhead_pred_ns.append(rec.get("pred_ns_per_step", 0))
        if last is not None:
            final_records.append(last)
    if not final_records:
        return {}
    return {
        "n_runs": len(final_records),
        "final_n_steps_mean": round(
            statistics.mean(r["n_steps"] for r in final_records), 0
        ),
        "final_pred_exact_rate_mean": round(
            statistics.mean(r["pred_exact_rate"] for r in final_records), 4
        ),
        "final_pred_dist_p50_mean": round(
            statistics.mean(r["pred_dist_p50"] for r in final_records), 1
        ),
        "final_pred_dist_p99_mean": round(
            statistics.mean(r["pred_dist_p99"] for r in final_records), 1
        ),
        "ramp_share_pct": round(
            100.0 * statistics.mean(
                r["n_predicted_ramp"] / max(1, r["n_steps"])
                for r in final_records
            ), 2
        ),
        "obs_ns_per_step_mean": round(
            statistics.mean(overhead_obs_ns), 1
        ) if overhead_obs_ns else None,
        "pred_ns_per_step_mean": round(
            statistics.mean(overhead_pred_ns), 1
        ) if overhead_pred_ns else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        default=os.path.join(
            os.path.dirname(__file__), "runs"
        ),
    )
    ap.add_argument(
        "--v0-prefix",
        default="V0_vanilla",
        help="prefix for V0 (vanilla) runs",
    )
    ap.add_argument(
        "--v1-prefix",
        default="V1_observe",
        help="prefix for V1 (hook on) runs",
    )
    args = ap.parse_args()

    v0 = _load_runs(args.runs, args.v0_prefix)
    v1 = _load_runs(args.runs, args.v1_prefix)

    print(f"=== L12 aggregate from {args.runs} ===\n")
    print(f"V0 vanilla runs:  {[r['_tag'] for r in v0]}")
    print(f"V1 observe runs:  {[r['_tag'] for r in v1]}")
    print()

    s0 = _summarise(v0)
    s1 = _summarise(v1)

    print(f"## throughput\n")
    print(f"  V0  output_tps: mean={s0.get('_tps_mean')}  std={s0.get('_tps_std')}")
    print(f"  V1  output_tps: mean={s1.get('_tps_mean')}  std={s1.get('_tps_std')}")
    if s0.get("_tps_mean") and s1.get("_tps_mean"):
        delta = (s1["_tps_mean"] - s0["_tps_mean"]) / s0["_tps_mean"] * 100
        print(f"  Δ V1 vs V0:  {delta:+.2f}%")
    print()

    print(f"## TTFT (ms) per phase — mean across runs (std)\n")
    print(f"  {'phase':<8} {'metric':<10}  {'V0':>16}  {'V1':>16}  {'Δ%':>8}")
    for ph in ("warm", "burst", "steady", "cool"):
        if ph not in s0 and ph not in s1:
            continue
        for k in ("ttft_p50", "ttft_p90", "ttft_p99", "ttft_mean"):
            v0_st = s0.get(ph, {}).get(k)
            v1_st = s1.get(ph, {}).get(k)
            if not v0_st or not v1_st:
                continue
            v0_fmt = f"{v0_st['mean']:.1f}±{v0_st['std']:.1f}"
            v1_fmt = f"{v1_st['mean']:.1f}±{v1_st['std']:.1f}"
            delta = (
                (v1_st["mean"] - v0_st["mean"]) / v0_st["mean"] * 100
                if v0_st["mean"] else None
            )
            d_fmt = f"{delta:+.1f}%" if delta is not None else "n/a"
            print(f"  {ph:<8} {k:<10}  {v0_fmt:>16}  {v1_fmt:>16}  {d_fmt:>8}")
        print()

    print("## predictor (V1 only)\n")
    p = _load_predictor(args.runs, "V1")
    if p:
        for k, v in p.items():
            print(f"  {k:<32}: {v}")
    else:
        print("  no predictor logs found")


if __name__ == "__main__":
    main()
