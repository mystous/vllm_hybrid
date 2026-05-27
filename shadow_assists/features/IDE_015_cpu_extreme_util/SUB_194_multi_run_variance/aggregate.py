#!/usr/bin/env python3
"""SUB_194 aggregate — multi-run mean ± stddev for AGSD-gated cells across Top-3 levers."""
from __future__ import annotations
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
LEVERS = {"L183": "SUB_183 NUMA pin",
          "L188": "SUB_188 softmax precompute",
          "L190": "SUB_190 tokenize worker"}
MIXES = ["balanced", "sonnet-heavy", "code-heavy"]


def load_tps(out_dir: Path, mix: str) -> float | None:
    p = out_dir / mix / f"benchmark_{mix}.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text())
    except Exception:
        return None
    for sc in j.get("scenarios", []):
        if sc.get("scenario") == "agsd-gated":
            return float(sc.get("tps") or 0) or None
    return None


def main() -> None:
    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    w("# SUB_194 — multi-run variance verification (Top-3 levers)")
    w()
    w("AGSD-gated only (vanilla-only / trident-only skipped). 3 runs per (lever, mode, mix) cell.")
    w()

    # global single 1-run reference from prior SUBs
    prior_delta = {"L183": 1.54, "L188": 1.84, "L190": 1.66}
    summary_rows = []

    for lever, lname in LEVERS.items():
        w(f"## {lever} — {lname}")
        w()
        w("| mix | OFF run1 | OFF run2 | OFF run3 | OFF mean ± sd | ON run1 | ON run2 | ON run3 | ON mean ± sd | Δ% mean | Δ% sd |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        per_mix_delta = []
        per_mix_sd = []
        agsd_off_means = []
        agsd_on_means = []
        for mix in MIXES:
            off_runs = []
            on_runs = []
            for r in (1, 2, 3):
                v_off = load_tps(BASE / "measurements" / lever / "off" / f"run{r}", mix)
                v_on = load_tps(BASE / "measurements" / lever / "on" / f"run{r}", mix)
                if v_off:
                    off_runs.append(v_off)
                if v_on:
                    on_runs.append(v_on)

            def fmt(vals: list[float]) -> str:
                if not vals:
                    return "—"
                m = statistics.mean(vals)
                if len(vals) >= 2:
                    sd = statistics.stdev(vals)
                    return f"{m:.1f} ± {sd:.1f}"
                return f"{m:.1f}"

            def vfmt(v: float | None) -> str:
                return "—" if v is None else f"{v:.1f}"

            off_m = statistics.mean(off_runs) if off_runs else None
            on_m = statistics.mean(on_runs) if on_runs else None
            off_sd = statistics.stdev(off_runs) if len(off_runs) >= 2 else None
            on_sd = statistics.stdev(on_runs) if len(on_runs) >= 2 else None

            if off_m and on_m:
                # per-run delta percentages
                deltas = []
                for ov, nv in zip(off_runs, on_runs):
                    if ov:
                        deltas.append((nv - ov) / ov * 100)
                d_mean = statistics.mean(deltas) if deltas else None
                d_sd = statistics.stdev(deltas) if len(deltas) >= 2 else None
                per_mix_delta.append(d_mean)
                per_mix_sd.append(d_sd)
                agsd_off_means.append(off_m)
                agsd_on_means.append(on_m)
            else:
                d_mean = None
                d_sd = None

            run_off = list(off_runs) + [None] * (3 - len(off_runs))
            run_on = list(on_runs) + [None] * (3 - len(on_runs))
            w("| {mix} | {o1} | {o2} | {o3} | {om} | {n1} | {n2} | {n3} | {nm} | {dm} | {dsd} |".format(
                mix=mix,
                o1=vfmt(run_off[0]), o2=vfmt(run_off[1]), o3=vfmt(run_off[2]),
                om=fmt(off_runs),
                n1=vfmt(run_on[0]), n2=vfmt(run_on[1]), n3=vfmt(run_on[2]),
                nm=fmt(on_runs),
                dm=("—" if d_mean is None else f"{d_mean:+.2f}%"),
                dsd=("—" if d_sd is None else f"{d_sd:.2f}"),
            ))

        # 3-mix avg delta
        valid = [d for d in per_mix_delta if d is not None]
        valid_sd = [s for s in per_mix_sd if s is not None]
        if valid:
            agg_delta = statistics.mean(valid)
            # propagate stddev across mixes if available
            if valid_sd:
                # pooled sd across the 3 mixes (sqrt of average variance)
                pooled = math.sqrt(statistics.mean(s * s for s in valid_sd))
            else:
                pooled = None
            w()
            w(f"**3-mix avg Δ% mean = {agg_delta:+.2f}%**  (per-run pooled sd ≈ {('%.2f' % pooled) if pooled is not None else '—'}, |sd|/|Δ| = {(pooled/abs(agg_delta) if agg_delta else float('nan')):.2f} if Δ≠0)")
            # verdict
            prior = prior_delta.get(lever)
            if pooled is not None and abs(agg_delta) > 1e-6:
                ratio = pooled / abs(agg_delta)
                if ratio < 0.5:
                    verdict = "BINDING (sd < magnitude/2)"
                elif ratio < 1.0:
                    verdict = "weak (magnitude/2 ≤ sd < magnitude)"
                else:
                    verdict = "NOISE (sd ≥ magnitude)"
            else:
                verdict = "insufficient runs"
            w()
            w(f"prior 1-run Δ = {prior:+.2f}% (reference) → multi-run Δ = {agg_delta:+.2f}% → verdict: **{verdict}**")
            summary_rows.append((lever, lname, prior, agg_delta, pooled, verdict))
        w()

    # final summary
    w("## Top-3 multi-run summary")
    w()
    w("| lever | name | 1-run Δ% (prior) | 3-run Δ% mean | pooled sd | verdict |")
    w("|---|---|---:|---:|---:|---|")
    for lever, lname, prior, m, sd, verdict in summary_rows:
        w(f"| {lever} | {lname} | {prior:+.2f}% | {m:+.2f}% | {('%.2f' % sd) if sd is not None else '—'} | {verdict} |")
    w()

    (BASE / "AGGREGATE.md").write_text("\n".join(lines))
    print(f"[saved] {BASE / 'AGGREGATE.md'}")


if __name__ == "__main__":
    main()
