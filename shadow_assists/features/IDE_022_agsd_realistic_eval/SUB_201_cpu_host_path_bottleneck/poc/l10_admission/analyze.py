"""[SUB_201/L10] aggregate seed runs → table snippets for MEASUREMENTS.md."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
import sys

ROOT = Path(
    "/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/"
    "SUB_201_cpu_host_path_bottleneck/poc/l10_admission"
)


def load_runs(subdir: str, tag: str) -> list[dict]:
    p = ROOT / subdir
    out = []
    for f in sorted(p.glob(f"{tag}_s*.json")):
        if ".invalid." in f.name:
            continue
        rec = json.loads(f.read_text())
        # skip runs where the server died early
        if rec.get("n_ok", 0) == 0 or rec.get("overall", {}).get("ttft_ms_p50") is None:
            continue
        out.append(rec)
    if out:
        return out
    # fall back to unsuffixed single-seed files
    f_plain = p / f"{tag}.json"
    if f_plain.exists():
        out.append(json.loads(f_plain.read_text()))
    return out


def agg(runs: list[dict], section: str, metric: str) -> tuple[float, float]:
    xs = [r[section][metric] for r in runs if r[section][metric] is not None]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt(mean: float, std: float, unit: str = "") -> str:
    if std == 0:
        return f"{mean:.1f}{unit}"
    return f"{mean:.1f}±{std:.1f}{unit}"


def pct_delta(a: float, b: float) -> str:
    if a == 0:
        return "—"
    return f"{(b - a) / a * 100:+.1f}%"


def render(subdir: str, label: str) -> str:
    bl = load_runs(subdir, "BASELINE")
    ba = load_runs(subdir, "BURSTAWARE")
    if not bl or not ba:
        return f"### {label}: missing runs (bl={len(bl)} ba={len(ba)})\n"

    lines = [f"### {label}  (mean±std over {len(bl)} seed runs)\n"]
    for section in ("overall", "short", "long"):
        n = bl[0][section]["n"]
        lines.append(f"\n**{section}** (n={n} per run)\n")
        lines.append("| metric | BASELINE | BURSTAWARE | Δ% |")
        lines.append("|---|---:|---:|---:|")
        for metric in (
            "ttft_ms_p50",
            "ttft_ms_p90",
            "ttft_ms_p99",
            "tpot_ms_p50",
            "tpot_ms_p99",
        ):
            bm, bs = agg(bl, section, metric)
            am, as_ = agg(ba, section, metric)
            lines.append(
                f"| {metric} | {fmt(bm, bs)} | {fmt(am, as_)} | {pct_delta(bm, am)} |"
            )
    # also wall + n_ok per run
    walls_bl = [r["wall_total_s"] for r in bl]
    walls_ba = [r["wall_total_s"] for r in ba]
    lines.append("\n**run-level**\n")
    lines.append("| metric | BASELINE | BURSTAWARE |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| wall_total_s | {statistics.mean(walls_bl):.1f}±{statistics.stdev(walls_bl) if len(walls_bl)>1 else 0:.1f} | {statistics.mean(walls_ba):.1f}±{statistics.stdev(walls_ba) if len(walls_ba)>1 else 0:.1f} |"
    )
    ok_bl = [r["n_ok"] for r in bl]
    ok_ba = [r["n_ok"] for r in ba]
    lines.append(f"| n_ok (avg) | {sum(ok_bl)/len(ok_bl):.0f} | {sum(ok_ba)/len(ok_ba):.0f} |")
    return "\n".join(lines) + "\n"


def main():
    out = []
    out.append("# L10 aggregate (auto-generated)\n")
    if (ROOT / "runs").exists():
        out.append(render("runs", "light load (1-seed, idle_mean=0.6s)"))
    if (ROOT / "runs_heavy").exists():
        out.append(render("runs_heavy", "heavy load (3-seed, idle_mean=0.15s)"))
    text = "\n".join(out)
    print(text)
    # Also dump to a fragment file the MEASUREMENTS template can pull in.
    (ROOT / "_agg.md").write_text(text)


if __name__ == "__main__":
    main()
