#!/usr/bin/env python3
"""
compare.py — N개 벤치마크 결과 비교

Usage:
    python compare.py <result_dir1> <result_dir2> [result_dir3 ...]
    python compare.py results/20260407_*  (glob)

Output:
    Console에 비교 테이블 출력
    첫 번째 결과 디렉토리에 comparison.txt / comparison.json 저장

Examples:
    python compare.py results/20260407_175241_* results/20260407_180242_*
    python compare.py results/20260407_1[5-8]*
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# Benchmark metrics
# ---------------------------------------------------------------------------

BENCH_KEYS = {
    "wall_time_s":            ("Wall Time (s)", "lower_better", ".1f"),
    "request_throughput":     ("Req TP (req/s)", "higher_better", ".2f"),
    "output_throughput":      ("Out TP (tok/s)", "higher_better", ".0f"),
    "total_token_throughput": ("Total TP (tok/s)", "higher_better", ".0f"),
    "duration":               ("Duration (s)", "lower_better", ".1f"),
    "mean_ttft_ms":           ("Mean TTFT (ms)", "lower_better", ".1f"),
    "median_ttft_ms":         ("Median TTFT (ms)", "lower_better", ".1f"),
    "p99_ttft_ms":            ("P99 TTFT (ms)", "lower_better", ".1f"),
    "mean_tpot_ms":           ("Mean TPOT (ms)", "lower_better", ".2f"),
    "median_tpot_ms":         ("Median TPOT (ms)", "lower_better", ".2f"),
    "p99_tpot_ms":            ("P99 TPOT (ms)", "lower_better", ".2f"),
    "mean_itl_ms":            ("Mean ITL (ms)", "lower_better", ".2f"),
    "median_itl_ms":          ("Median ITL (ms)", "lower_better", ".2f"),
    "p99_itl_ms":             ("P99 ITL (ms)", "lower_better", ".2f"),
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict | None:
    """Load a single run directory. Auto-detect gpu_only.json or hybrid.json."""
    run = {"dir": run_dir, "name": run_dir.name}

    # Load benchmark JSON (prefer hybrid, fallback to gpu_only)
    for mode in ("hybrid", "gpu_only"):
        jf = run_dir / f"{mode}.json"
        if jf.exists():
            with open(jf) as f:
                run["bench"] = json.load(f)
            run["mode"] = mode
            break

    if "bench" not in run:
        return None

    # Load system_info
    si_path = run_dir / "system_info.json"
    if si_path.exists():
        with open(si_path) as f:
            run["system_info"] = json.load(f)
    else:
        run["system_info"] = {}

    # Load monitor CSVs
    mode = run["mode"]
    for mtype in ("gpu", "cpu"):
        csv_path = run_dir / f"{mode}_monitor_{mtype}.csv"
        run[f"monitor_{mtype}"] = _summarize_csv(csv_path)

    return run


def _summarize_csv(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    try:
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            return None
        headers = list(rows[0].keys())
        util_cols = [h for h in headers if h.endswith("_util_pct") or h.endswith("_power_w")]
        summary = {}
        for col in util_cols:
            vals = [float(r[col]) for r in rows if r.get(col, "") not in ("", "N/A")]
            if vals:
                summary[col] = {
                    "mean": round(sum(vals) / len(vals), 2),
                    "max": round(max(vals), 2),
                    "min": round(min(vals), 2),
                }
        summary["sample_count"] = len(rows)
        summary["duration_s"] = round(
            float(rows[-1]["elapsed_s"]) - float(rows[0]["elapsed_s"]), 1
        ) if len(rows) > 1 else 0.0
        return summary
    except Exception as e:
        print(f"[WARN] CSV parse failed ({csv_path.name}): {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def make_label(run: dict) -> str:
    """Short label for a run: mode + routing + model."""
    si = run.get("system_info", {})
    hc = si.get("hybrid_config", {})
    bench = run.get("bench", {})

    mode = run.get("mode", "?")
    model = bench.get("model_id", "?").split("/")[-1]

    if mode == "gpu_only":
        return f"gpu_only | {model}"

    strategy = hc.get("routing_strategy", "?")
    priority = hc.get("routing_priority", "?")
    seqs = hc.get("cpu_max_seqs", "?")

    if strategy == "round-robin":
        routing = "RR"
    else:
        routing = f"{strategy}({priority})"

    return f"{mode} {routing} seqs={seqs} | {model}"


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(runs: list[dict]) -> tuple[str, dict]:
    lines = []
    result_json = {
        "generated_at": datetime.now(KST).isoformat(),
        "runs": [],
    }

    # Header
    lines.append("=" * 100)
    lines.append("  vLLM Benchmark Comparison Report")
    lines.append(f"  Generated: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}")
    lines.append(f"  Comparing {len(runs)} runs")
    lines.append("=" * 100)
    lines.append("")

    # Run info table
    labels = []
    for i, r in enumerate(runs):
        label = make_label(r)
        tag = f"[{i}]"
        labels.append((tag, label))

        bench = r.get("bench", {})
        completed = bench.get("completed", "?")
        prompts = bench.get("num_prompts", "?")
        lines.append(f"  {tag} {label}")
        lines.append(f"       Dir: {r['name']}  |  {completed}/{prompts} completed")

        # Save to JSON
        si = r.get("system_info", {})
        gpu_devs = si.get("gpu", {}).get("devices", [{}])
        gpu_name = gpu_devs[0].get("name", "-") if gpu_devs else "-"
        result_json["runs"].append({
            "index": i,
            "label": label,
            "dir": r["name"],
            "mode": r.get("mode", "?"),
            "hybrid_config": si.get("hybrid_config", {}),
            "gpu": gpu_name,
            "cpu": si.get("cpu", {}).get("model_name", "-"),
        })

    lines.append("")

    # Metrics table — values
    n = len(runs)
    col_w = 14

    lines.append("-" * 100)
    hdr = f"  {'Metric':<24}"
    for tag, _ in labels:
        hdr += f" {tag:>{col_w}}"
    lines.append(hdr)
    lines.append("-" * 100)

    result_json["metrics"] = {}

    for key, (label, direction, fmt) in BENCH_KEYS.items():
        vals = [r["bench"].get(key) for r in runs]
        if all(v is None for v in vals):
            continue

        row = f"  {label:<24}"
        for v in vals:
            row += f" {format(v, fmt):>{col_w}}" if v is not None else f" {'N/A':>{col_w}}"
        lines.append(row)

        result_json["metrics"][key] = {
            "label": label, "direction": direction,
            "values": [{"run": i, "value": v} for i, v in enumerate(vals)],
        }

    lines.append("-" * 100)
    lines.append("")

    # Diff vs [0] — each run on its own row
    if n >= 2:
        lines.append("  [vs [0] Comparison]")
        base = runs[0]["bench"]

        for i in range(1, n):
            tag = labels[i][0]
            lines.append(f"  {tag} {labels[i][1]}")
            comp = runs[i]["bench"]

            for key, (label, direction, fmt) in BENCH_KEYS.items():
                bv = base.get(key)
                cv = comp.get(key)
                if bv is None or cv is None:
                    continue
                if bv == 0:
                    continue
                diff_pct = (cv - bv) / abs(bv) * 100
                if direction == "higher_better":
                    marker = "▲" if diff_pct > 1 else ("▼" if diff_pct < -1 else "~")
                else:
                    marker = "▼" if diff_pct > 1 else ("▲" if diff_pct < -1 else "~")
                lines.append(
                    f"    {label:<24} {format(bv, fmt):>10} → {format(cv, fmt):>10}"
                    f"  {diff_pct:+6.1f}% {marker}"
                )
            lines.append("")
        lines.append("-" * 100)
        lines.append("")

    # Key summary
    lines.append("  [Key Summary]")
    for i, r in enumerate(runs):
        b = r["bench"]
        tag = labels[i][0]
        req = b.get("request_throughput", 0)
        tok = b.get("output_throughput", 0)
        ttft = b.get("mean_ttft_ms", 0)
        wall = b.get("wall_time_s")
        wall_str = f"  |  wall {wall:.1f}s" if wall else ""
        lines.append(f"  {tag} {req:.2f} req/s  |  {tok:.0f} tok/s  |  TTFT {ttft:.0f}ms{wall_str}")
    lines.append("")

    # GPU utilization summary
    lines.append("-" * 100)
    lines.append("  [GPU Utilization]")
    for i, r in enumerate(runs):
        tag = labels[i][0]
        mon = r.get("monitor_gpu")
        if not mon:
            lines.append(f"  {tag} No GPU monitor data")
            continue
        avg_util = mon.get("gpu_avg_util_pct", {})
        avg_power = mon.get("gpu_avg_power_w", {})
        util_str = f"avg={avg_util.get('mean', 0):.1f}%" if avg_util else "N/A"
        power_str = f"avg={avg_power.get('mean', 0):.0f}W" if avg_power else "N/A"
        lines.append(
            f"  {tag} {util_str}  max={avg_util.get('max', 0):.0f}%"
            f"  |  Power {power_str}"
            f"  ({mon.get('sample_count', 0)} samples, {mon.get('duration_s', 0):.0f}s)"
        )
    lines.append("")

    # CPU utilization summary
    lines.append("  [CPU Utilization]")
    for i, r in enumerate(runs):
        tag = labels[i][0]
        mon = r.get("monitor_cpu")
        if not mon:
            lines.append(f"  {tag} No CPU monitor data")
            continue
        avg_util = mon.get("cpu_avg_util_pct", {})
        util_str = f"avg={avg_util.get('mean', 0):.1f}%" if avg_util else "N/A"
        lines.append(
            f"  {tag} {util_str}  max={avg_util.get('max', 0):.0f}%"
            f"  ({mon.get('sample_count', 0)} samples, {mon.get('duration_s', 0):.0f}s)"
        )
    lines.append("")

    # Power efficiency
    lines.append("  [Power Efficiency]")
    for i, r in enumerate(runs):
        tag = labels[i][0]
        mon = r.get("monitor_gpu")
        tok = r["bench"].get("output_throughput", 0)
        if not mon or not tok:
            lines.append(f"  {tag} N/A")
            continue
        avg_power = mon.get("gpu_avg_power_w", {}).get("mean", 0)
        if avg_power > 0:
            tok_per_w = tok / avg_power
            dur = r["bench"].get("duration", 0)
            total_energy_wh = avg_power * dur / 3600
            total_tokens = r["bench"].get("total_output_tokens", 0)
            tok_per_wh = total_tokens / total_energy_wh if total_energy_wh > 0 else 0
            lines.append(
                f"  {tag} {avg_power:.0f}W avg"
                f"  |  {tok_per_w:.1f} tok/s/W"
                f"  |  {total_energy_wh:.1f} Wh consumed"
                f"  |  {tok_per_wh:.0f} tok/Wh"
            )
        else:
            lines.append(f"  {tag} No power data")
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines), result_json


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare N benchmark results",
        usage="%(prog)s <result_dir1> <result_dir2> [...]",
    )
    parser.add_argument(
        "dirs", nargs="+", metavar="RESULT_DIR",
        help="Result directories to compare",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory to save comparison files (default: first result dir)",
    )
    args = parser.parse_args()

    # Load runs
    runs = []
    for d in args.dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"[WARN] Not a directory, skipping: {d}", file=sys.stderr)
            continue
        run = load_run(p)
        if run is None:
            print(f"[WARN] No benchmark JSON found, skipping: {d}", file=sys.stderr)
            continue
        runs.append(run)

    if len(runs) < 1:
        print("[ERROR] No valid result directories found.", file=sys.stderr)
        sys.exit(1)

    if len(runs) == 1:
        print("[INFO] Only 1 run found. Showing summary (no comparison).")

    # Build report
    report_txt, report_json = build_report(runs)
    print(report_txt)

    # Save
    out_dir = Path(args.output_dir) if args.output_dir else runs[0]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "comparison.txt"
    json_path = out_dir / "comparison.json"

    txt_path.write_text(report_txt)
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    print(f"\n[compare] Saved:")
    print(f"  Text → {txt_path}")
    print(f"  JSON → {json_path}")


if __name__ == "__main__":
    main()
