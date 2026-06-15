#!/usr/bin/env python3
"""TSK_042 unified 10-model LHC Path 1 validation — aggregate all.

Inputs:
  lhc_phase4/tsk042_10model_unified/<TAG>/{vanilla,lhc_path1}_runs/<corpus>_s<sw>.json

Outputs:
  - <ROOT>/TSK042_UNIFIED_FINAL.md  (10 × 7 corpus Δ% matrix + verdicts)
  - <ROOT>/aggregate_all.csv
  - <ROOT>/aggregate_all.json
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO = Path("/workspace/host_vllm_hybrid")
ROOT = REPO / "lhc_phase4/tsk042_10model_unified"

MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Llama-3.1-70B-Instruct",
    "Qwen2.5-32B-Instruct",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Llama-70B",
    "Qwen2.5-72B-Instruct",
    "Llama-3.1-405B-Instruct-FP8",
    "DeepSeek-R1",
]
CORPORA = ["sharegpt", "swebench", "humaneval", "mbpp", "wildchat", "lmsys", "mix"]
CONFIGS = ["vanilla", "lhc_path1"]
METRIC = "output_tps"


def load_runs(base: Path, cfg: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {c: [] for c in CORPORA}
    d = base / f"{cfg}_runs"
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        stem = p.stem
        if "_s" not in stem:
            continue
        corpus, sw = stem.rsplit("_s", 1)
        if corpus not in out:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        data["_sweep"] = int(sw)
        out[corpus].append(data)
    for c in out:
        out[c].sort(key=lambda r: r["_sweep"])
    return out


def paired_delta_ci(va: list[float], lh: list[float]):
    n = min(len(va), len(lh))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    pairs = [(va[i], lh[i]) for i in range(n)]
    mva = sum(p[0] for p in pairs) / n
    mlh = sum(p[1] for p in pairs) / n
    if mva == 0:
        return mva, mlh, float("nan"), float("nan"), n
    deltas = [(p[1] - p[0]) / p[0] * 100.0 for p in pairs]
    md = sum(deltas) / n
    if n < 2:
        return mva, mlh, md, float("nan"), n
    sd = statistics.stdev(deltas)
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
               7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    df = n - 1
    t = t_table.get(df, 1.96)
    halfw = t * sd / math.sqrt(n)
    return mva, mlh, md, halfw, n


def main() -> None:
    matrix: dict[str, dict[str, dict]] = {}
    for tag in MODELS:
        base = ROOT / tag
        rows: dict[str, dict] = {}
        va = load_runs(base, "vanilla")
        lh = load_runs(base, "lhc_path1")
        for c in CORPORA:
            va_tps = [r.get(METRIC) for r in va.get(c, []) if r.get(METRIC) is not None]
            lh_tps = [r.get(METRIC) for r in lh.get(c, []) if r.get(METRIC) is not None]
            mva, mlh, dpct, halfw, n = paired_delta_ci(va_tps, lh_tps)
            rows[c] = dict(n=n, vanilla=mva, lhc=mlh, delta_pct=dpct, ci95=halfw)
        matrix[tag] = rows

    (ROOT / "aggregate_all.json").write_text(json.dumps(matrix, indent=2, default=str))
    rows = ["model,corpus,n,vanilla_tps,lhc_tps,delta_pct,ci95_half_pct"]
    for tag in MODELS:
        for c in CORPORA:
            r = matrix[tag][c]
            rows.append(
                f"{tag},{c},{r['n']},{r['vanilla']:.2f},{r['lhc']:.2f},{r['delta_pct']:.2f},{r['ci95']:.2f}"
            )
    (ROOT / "aggregate_all.csv").write_text("\n".join(rows) + "\n")

    md = ["# TSK_042 10-model LHC Path 1 Unified Validation — Final\n",
          "Generated: aggregate_all.py (single sequential agent)\n",
          "## Δ% Matrix (LHC Path 1 vs vanilla — paired by sweep)\n",
          "Cell format: `Δ%±CI95 (n)`. `—` = no data.\n",
          ""]
    md.append("| Model | " + " | ".join(CORPORA) + " |")
    md.append("|---" + "|---" * len(CORPORA) + "|")
    pos_cells: list[tuple[str, str, float, float]] = []
    for tag in MODELS:
        cells = []
        for c in CORPORA:
            r = matrix[tag][c]
            if r["n"] == 0 or math.isnan(r["delta_pct"]):
                cells.append("—")
            else:
                ci = f"±{r['ci95']:.1f}" if not math.isnan(r["ci95"]) else ""
                cells.append(f"{r['delta_pct']:+.1f}{ci} ({r['n']})")
                if r["delta_pct"] >= 5.0 and (
                    math.isnan(r["ci95"]) or r["delta_pct"] - r["ci95"] > 0
                ):
                    pos_cells.append((tag, c, r["delta_pct"], r["ci95"]))
        md.append(f"| {tag} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Positive Cells (Δ% ≥ +5%, CI95 LB > 0)\n")
    if not pos_cells:
        md.append("- (none yet)\n")
    else:
        for tag, c, d, hw in pos_cells:
            md.append(f"- **{tag} x {c}**: Δ=`{d:+.2f}%` ±{hw:.2f}%")
    md.append("")
    md.append("## Notes\n")
    md.append("- All 10 models run in a single sequential agent (no concurrent agent contention).\n")
    md.append("- Sweep schedule per user spec: 8B/7B=3, 32B=3, 70B/72B=2, 405B=1, R1=1 (mix only).\n")
    md.append("- Same harness as TSK_042 (concurrency=32, max_tokens=8192, streaming).\n")
    md.append("- LHC Path 1: VLLM_LHC_AMX_C3_PREFIX=1 + libamx_c3.so.\n")

    (ROOT / "TSK042_UNIFIED_FINAL.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {ROOT / 'aggregate_all.csv'}")
    print(f"Wrote {ROOT / 'aggregate_all.json'}")
    print(f"Wrote {ROOT / 'TSK042_UNIFIED_FINAL.md'}")


if __name__ == "__main__":
    main()
