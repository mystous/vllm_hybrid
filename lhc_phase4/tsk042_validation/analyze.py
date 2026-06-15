#!/usr/bin/env python3
"""TSK_042 LHC Path 1 validation — aggregate 7 corpus × 2 config × N sweeps.

Inputs:
  lhc_phase4/tsk042_validation/vanilla_runs/<corpus>_s<sw>.json
  lhc_phase4/tsk042_validation/lhc_path1_runs/<corpus>_s<sw>.json

Outputs:
  - prints corpus × Δ% table + 95% CI (paired)
  - dumps lhc_phase4/tsk042_validation/aggregate.json + per_corpus_table.csv
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path("/workspace/host_vllm_hybrid/lhc_phase4/tsk042_validation")
CORPORA = ["sharegpt", "swebench", "humaneval", "mbpp", "wildchat", "lmsys", "mix"]
CONFIGS = ["vanilla", "lhc_path1"]
METRIC = "output_tps"


def load_runs(cfg: str) -> dict[str, list[dict]]:
    """corpus -> list[run summary] sorted by sweep #."""
    base = ROOT / f"{cfg}_runs"
    out: dict[str, list[dict]] = {c: [] for c in CORPORA}
    if not base.exists():
        return out
    for p in sorted(base.glob("*.json")):
        # filename: <corpus>_s<N>.json
        stem = p.stem
        if "_s" not in stem:
            continue
        corpus, sw = stem.rsplit("_s", 1)
        if corpus not in out:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed to read {p}: {exc}")
            continue
        data["_sweep"] = int(sw)
        data["_file"] = str(p)
        out[corpus].append(data)
    for c in out:
        out[c].sort(key=lambda r: r["_sweep"])
    return out


def paired_delta_ci(va: list[float], lh: list[float]) -> tuple[float, float, float, float]:
    """Returns (mean_va, mean_lh, delta_pct_mean, ci95_halfwidth_pct).

    Paired: delta_i = (lh_i - va_i) / va_i * 100. n=len(va)=len(lh).
    95% CI half = t_{n-1, .975} * sd / sqrt(n). For small n use t-table approximation.
    """
    n = min(len(va), len(lh))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pairs = [(va[i], lh[i]) for i in range(n)]
    mva = sum(p[0] for p in pairs) / n
    mlh = sum(p[1] for p in pairs) / n
    if mva == 0:
        return mva, mlh, float("nan"), float("nan")
    deltas = [(p[1] - p[0]) / p[0] * 100.0 for p in pairs]
    md = sum(deltas) / n
    if n < 2:
        return mva, mlh, md, float("nan")
    sd = statistics.stdev(deltas)
    # t-critical 95% two-sided
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
               7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    df = n - 1
    t = t_table.get(df, 1.96)
    halfw = t * sd / math.sqrt(n)
    return mva, mlh, md, halfw


def main() -> None:
    runs = {cfg: load_runs(cfg) for cfg in CONFIGS}
    print(f"Loaded runs:")
    for cfg, mp in runs.items():
        for c, lst in mp.items():
            tps = [r.get(METRIC) for r in lst]
            print(f"  {cfg:10s} {c:10s} n={len(lst):2d} sweeps={[r['_sweep'] for r in lst]} {METRIC}={tps}")
    print()

    # Build paired table
    headers = ["corpus", "n", "vanilla_tps", "lhc_tps", "delta_pct", "ci95_half_pct",
               "prefix_cache_hit_pct_va", "prefix_cache_hit_pct_lh"]
    print("-" * 110)
    print(f"{'corpus':<10s} {'n':>3s} {'vanilla':>10s} {'lhc_p1':>10s} {'Δ%':>8s} {'±CI95':>8s} "
          f"{'ttft_va':>9s} {'ttft_lh':>9s} {'tpot_va':>9s} {'tpot_lh':>9s}")
    print("-" * 110)
    csv_rows = [",".join(headers + ["ttft_p50_va", "ttft_p50_lh", "tpot_p50_va", "tpot_p50_lh"])]
    summary = {}
    pos_corpora = []
    for c in CORPORA:
        va_runs = runs["vanilla"].get(c, [])
        lh_runs = runs["lhc_path1"].get(c, [])
        n = min(len(va_runs), len(lh_runs))
        if n == 0:
            print(f"{c:<10s} {0:>3d}   - no data -")
            continue
        va_tps = [va_runs[i].get(METRIC) for i in range(n)]
        lh_tps = [lh_runs[i].get(METRIC) for i in range(n)]
        va_tps = [x for x in va_tps if x is not None]
        lh_tps = [x for x in lh_tps if x is not None]
        n = min(len(va_tps), len(lh_tps))
        if n == 0:
            print(f"{c:<10s} {0:>3d}   - no valid tps -")
            continue
        mva, mlh, dpct, halfw = paired_delta_ci(va_tps[:n], lh_tps[:n])
        # TTFT / TPOT
        ttft_va = [r.get("ttft_ms_p50") for r in va_runs[:n] if r.get("ttft_ms_p50") is not None]
        ttft_lh = [r.get("ttft_ms_p50") for r in lh_runs[:n] if r.get("ttft_ms_p50") is not None]
        tpot_va = [r.get("tpot_ms_p50") for r in va_runs[:n] if r.get("tpot_ms_p50") is not None]
        tpot_lh = [r.get("tpot_ms_p50") for r in lh_runs[:n] if r.get("tpot_ms_p50") is not None]
        ttft_va_m = sum(ttft_va) / len(ttft_va) if ttft_va else float("nan")
        ttft_lh_m = sum(ttft_lh) / len(ttft_lh) if ttft_lh else float("nan")
        tpot_va_m = sum(tpot_va) / len(tpot_va) if tpot_va else float("nan")
        tpot_lh_m = sum(tpot_lh) / len(tpot_lh) if tpot_lh else float("nan")
        sig = "*" if not math.isnan(halfw) and abs(dpct) > halfw else " "
        flag = " ✓+5%" if dpct >= 5.0 and (math.isnan(halfw) or dpct - halfw > 0) else ""
        print(f"{c:<10s} {n:>3d} {mva:>10.1f} {mlh:>10.1f} {dpct:>+7.2f}{sig} {halfw:>7.2f} "
              f"{ttft_va_m:>9.1f} {ttft_lh_m:>9.1f} {tpot_va_m:>9.2f} {tpot_lh_m:>9.2f}{flag}")
        csv_rows.append(",".join(str(x) for x in [
            c, n, f"{mva:.1f}", f"{mlh:.1f}", f"{dpct:.2f}", f"{halfw:.2f}",
            "", "", f"{ttft_va_m:.1f}", f"{ttft_lh_m:.1f}",
            f"{tpot_va_m:.2f}", f"{tpot_lh_m:.2f}",
        ]))
        summary[c] = dict(
            n=n,
            vanilla_tps_mean=mva, lhc_tps_mean=mlh,
            delta_pct=dpct, ci95_half_pct=halfw,
            vanilla_tps_per_sweep=va_tps[:n], lhc_tps_per_sweep=lh_tps[:n],
            ttft_p50_va=ttft_va_m, ttft_p50_lh=ttft_lh_m,
            tpot_p50_va=tpot_va_m, tpot_p50_lh=tpot_lh_m,
            significant=(not math.isnan(halfw) and abs(dpct) > halfw),
        )
        if dpct >= 5.0 and (math.isnan(halfw) or dpct - halfw > 0):
            pos_corpora.append((c, dpct, halfw))
    print("-" * 110)
    print(f"(* = |Δ| > CI95 half;  ✓+5% = lower bound of CI95 above +5%)")
    print()
    if pos_corpora:
        print("✓ Positive corpora (LHC Path 1 ≥ +5%, CI95 LB > 0):")
        for c, d, hw in pos_corpora:
            print(f"    {c}: Δ={d:+.2f}% ±{hw:.2f}%")
    else:
        print("✗ No corpus reached the +5% / CI95-LB>0 threshold.")
    out_json = ROOT / "aggregate.json"
    out_json.write_text(json.dumps({"summary": summary,
                                    "positive_corpora": [c for c, _, _ in pos_corpora]}, indent=2))
    print(f"\nWrote {out_json}")
    (ROOT / "per_corpus_table.csv").write_text("\n".join(csv_rows) + "\n")
    print(f"Wrote {ROOT / 'per_corpus_table.csv'}")


if __name__ == "__main__":
    main()
