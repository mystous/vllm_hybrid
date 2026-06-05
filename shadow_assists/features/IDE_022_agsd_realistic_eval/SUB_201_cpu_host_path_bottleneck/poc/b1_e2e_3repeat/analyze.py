#!/usr/bin/env python3
"""B1 3-repeat × 4-run sweep analysis.

Reads 12 llama8b_r<rep>_<MODE>.json files, computes:
  - raw 12 measurement table
  - per-mode mean ± std (n=3 each)
  - Δ_B1 (C-A) / Δ_B3 (B-A) / Δ_B1+B3 (D-A) deltas with 95% CI
  - Mann-Whitney U test (non-parametric, n=3 per group)
  - paired t-test (parametric, same repeat indices)

Prints markdown sections suitable to paste into MEASUREMENTS.md.
"""
from __future__ import annotations

import json
import math
import statistics
from itertools import combinations
from pathlib import Path

POC_DIR = Path(__file__).parent
MODES = ["A_baseline", "B_b3", "C_b1", "D_b1b3"]
REPS = [1, 2, 3]

# t-critical for two-sided 95% CI, df = n-1
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def load_one(rep: int, mode: str) -> dict | None:
    p = POC_DIR / f"llama8b_r{rep}_{mode}.json"
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    d["_rep"] = rep
    d["_mode"] = mode
    return d


def mean_std(xs):
    if len(xs) < 2:
        return statistics.mean(xs) if xs else float("nan"), float("nan")
    return statistics.mean(xs), statistics.stdev(xs)


def ci95(xs):
    """95% CI half-width for the mean, t-distribution."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m, s = mean_std(xs)
    se = s / math.sqrt(n)
    tcrit = T_CRIT_95.get(n - 1, 1.96)
    return tcrit * se


def diff_ci95(a, b):
    """95% CI half-width for (mean(a) - mean(b)) using Welch's approximation."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    sea2 = sa * sa / na
    seb2 = sb * sb / nb
    se = math.sqrt(sea2 + seb2)
    num = (sea2 + seb2) ** 2
    den = (sea2 ** 2) / (na - 1) + (seb2 ** 2) / (nb - 1)
    if den == 0:
        df = na + nb - 2
    else:
        df = num / den
    # round df to nearest int >=1 for table lookup
    df_int = max(1, min(6, int(round(df))))
    tcrit = T_CRIT_95.get(df_int, 1.96)
    halfw = tcrit * se
    return ma - mb, halfw


def paired_diff_ci95(a_dict, b_dict):
    """Paired diff: same repeat indices. a/b dict {rep: tps}."""
    reps = sorted(set(a_dict) & set(b_dict))
    diffs = [a_dict[r] - b_dict[r] for r in reps]
    if len(diffs) < 2:
        return float("nan"), float("nan"), diffs
    m, s = mean_std(diffs)
    se = s / math.sqrt(len(diffs))
    tcrit = T_CRIT_95.get(len(diffs) - 1, 1.96)
    return m, tcrit * se, diffs


def paired_t_pvalue(diffs):
    """Two-sided paired t-test p-value, approximate via t-distribution survival.
    Uses scipy if available, else simple approximation."""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m, s = mean_std(diffs)
    if s == 0:
        return float("nan"), float("nan")
    t = m / (s / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore
        p = 2 * stats.t.sf(abs(t), df=n - 1)
        return t, p
    except Exception:
        # crude fallback: 2-sided based on T_CRIT_95
        tcrit = T_CRIT_95.get(n - 1, 1.96)
        p = 0.05 if abs(t) > tcrit else 0.5  # very crude
        return t, p


def mann_whitney_u(a, b):
    """Mann-Whitney U test, two-sided. Returns (U, p_approx)."""
    try:
        from scipy import stats  # type: ignore
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except Exception:
        # Exact for tiny n: rank-sum
        combined = sorted([(v, "a") for v in a] + [(v, "b") for v in b])
        ranks = {}
        i = 0
        while i < len(combined):
            j = i
            while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[k] = avg_rank
            i = j + 1
        ra = sum(ranks[k] for k, (_, src) in enumerate(combined) if src == "a")
        na, nb = len(a), len(b)
        u_a = ra - na * (na + 1) / 2
        u_b = na * nb - u_a
        u = min(u_a, u_b)
        # For n=3, n=3: exact 2-sided 5% threshold U≤0 (no significance possible with n=3)
        return u, float("nan")


def main():
    rows = []
    by_mode = {m: [] for m in MODES}
    by_mode_rep = {m: {} for m in MODES}
    for rep in REPS:
        for mode in MODES:
            d = load_one(rep, mode)
            if d is None:
                print(f"MISSING: r{rep}_{mode}")
                continue
            rows.append(d)
            by_mode[mode].append(d["output_tps"])
            by_mode_rep[mode][rep] = d["output_tps"]

    print("\n## Raw 12 measurements\n")
    print("| rep | mode | tps | n_ok | wall_s | TTFT p50 | TPOT p50 | GPU% | CPU% | err |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in sorted(rows, key=lambda x: (x["_rep"], MODES.index(x["_mode"]))):
        print(
            f"| {d['_rep']} | {d['_mode']} | {d['output_tps']:.1f} "
            f"| {d['n_ok']}/{d['n']} | {d['wall_total_s']:.1f} "
            f"| {d['ttft_ms_p50']:.1f} | {d['tpot_ms_p50']:.2f} "
            f"| {d.get('gpu_util', 0):.1f} | {d.get('cpu_util', 0):.1f} "
            f"| {d.get('n_err', 0)} |"
        )

    print("\n## Per-mode mean ± std (tps), n=3\n")
    print("| mode | n | mean | std | ci95_halfw | values |")
    print("|---|---:|---:|---:|---:|---|")
    for m in MODES:
        vals = by_mode[m]
        mu, sd = mean_std(vals)
        h = ci95(vals)
        vs = ", ".join(f"{v:.1f}" for v in vals)
        print(f"| {m} | {len(vals)} | {mu:.1f} | {sd:.1f} | ±{h:.1f} | {vs} |")

    print("\n## Δ analysis (vs A_baseline)\n")
    print("### Unpaired (Welch) — mean ± 95% CI half-width\n")
    print("| comparison | Δ_tps | 95% CI half-w | Δ% | CI cross 0? |")
    print("|---|---:|---:|---:|:---:|")
    a_vals = by_mode["A_baseline"]
    a_mean = statistics.mean(a_vals) if a_vals else float("nan")
    for m in ["B_b3", "C_b1", "D_b1b3"]:
        d_mean, halfw = diff_ci95(by_mode[m], a_vals)
        pct = 100 * d_mean / a_mean if a_mean else float("nan")
        crosses = "✗ sig" if (d_mean - halfw) * (d_mean + halfw) > 0 else "✓ NOT sig"
        print(
            f"| {m} − A | {d_mean:+.1f} | ±{halfw:.1f} | {pct:+.2f}% | {crosses} |"
        )

    print("\n### Paired (same repeat index) — mean Δ ± 95% CI half-width\n")
    print("| comparison | mean Δ | 95% CI half-w | Δ% | t-stat | p-value | per-rep Δ | sig? |")
    print("|---|---:|---:|---:|---:|---:|---|:---:|")
    for m in ["B_b3", "C_b1", "D_b1b3"]:
        m_diff, halfw, diffs = paired_diff_ci95(by_mode_rep[m], by_mode_rep["A_baseline"])
        t, p = paired_t_pvalue(diffs)
        pct = 100 * m_diff / a_mean if a_mean else float("nan")
        crosses = "✗ sig" if (m_diff - halfw) * (m_diff + halfw) > 0 else "✓ NOT sig"
        diffs_str = ", ".join(f"{x:+.1f}" for x in diffs)
        print(
            f"| {m} − A | {m_diff:+.1f} | ±{halfw:.1f} | {pct:+.2f}% "
            f"| {t:.2f} | {p:.4f} | {diffs_str} | {crosses} |"
        )

    print("\n### Mann-Whitney U test (non-parametric, n=3 each)\n")
    print("| comparison | U | p-value | NOTE |")
    print("|---|---:|---:|---|")
    for m in ["B_b3", "C_b1", "D_b1b3"]:
        u, p = mann_whitney_u(by_mode[m], a_vals)
        note = "n=3,3 → 2-sided exact p min = 0.10 (cannot reach 0.05)" if math.isnan(p) or True else ""
        # actually for n=3,3 exact, smallest 2-sided p is 0.10 — flag this
        print(f"| {m} vs A | {u:.1f} | {p:.4f} | {note} |")

    print("\n### Pairwise (B↔C, B↔D, C↔D) paired diff\n")
    print("| comparison | mean Δ | 95% CI half-w | sig? |")
    print("|---|---:|---:|:---:|")
    for a, b in [("B_b3", "C_b1"), ("B_b3", "D_b1b3"), ("C_b1", "D_b1b3")]:
        m_diff, halfw, diffs = paired_diff_ci95(by_mode_rep[a], by_mode_rep[b])
        crosses = "✗ sig" if (m_diff - halfw) * (m_diff + halfw) > 0 else "✓ NOT sig"
        print(f"| {a} − {b} | {m_diff:+.1f} | ±{halfw:.1f} | {crosses} |")


if __name__ == "__main__":
    main()
