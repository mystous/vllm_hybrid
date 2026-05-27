"""SUB_186 aggregate: extract OFF/ON tps per (mix, scenario) and compute Δ% vs SUB_182/SUB_183."""
import json
from pathlib import Path

BASE = Path(__file__).parent
MIXES = ["balanced", "sonnet-heavy", "code-heavy"]
SCENARIOS = ["vanilla-only", "trident-only", "agsd-gated"]


def load(mode: str, mix: str):
    p = BASE / "measurements" / mode / mix / f"benchmark_{mix}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    rows = []
    for mix in MIXES:
        off = load("off", mix)
        on = load("on", mix)
        for sc in SCENARIOS:
            off_tps = None
            on_tps = None
            if off:
                for s in off.get("scenarios", []):
                    if s["scenario"] == sc:
                        off_tps = s["tps"]
            if on:
                for s in on.get("scenarios", []):
                    if s["scenario"] == sc:
                        on_tps = s["tps"]
            delta = None
            if off_tps and on_tps:
                delta = (on_tps - off_tps) / off_tps * 100.0
            rows.append((mix, sc, off_tps, on_tps, delta))

    print("| mix | scenario | OFF tps | ON tps | Δ% |")
    print("|---|---|---:|---:|---:|")
    for mix, sc, off_tps, on_tps, delta in rows:
        off_s = f"{off_tps:,.1f}" if off_tps is not None else "—"
        on_s = f"{on_tps:,.1f}" if on_tps is not None else "—"
        d_s = f"{delta:+.2f}%" if delta is not None else "—"
        print(f"| {mix} | {sc} | {off_s} | {on_s} | {d_s} |")

    print()
    print("| scenario | 3-mix avg OFF | 3-mix avg ON | Δ% |")
    print("|---|---:|---:|---:|")
    agg = {}
    for sc in SCENARIOS:
        offs = [r[2] for r in rows if r[1] == sc and r[2] is not None]
        ons = [r[3] for r in rows if r[1] == sc and r[3] is not None]
        if len(offs) == 3 and len(ons) == 3:
            o = sum(offs) / 3
            n = sum(ons) / 3
            d = (n - o) / o * 100.0
            print(f"| {sc} | {o:,.1f} | {n:,.1f} | {d:+.2f}% |")
            agg[sc] = {"off": o, "on": n, "delta_pct": d}
        else:
            print(f"| {sc} | (incomplete) | (incomplete) | — |")
            agg[sc] = None

    # superposition comparison
    # SUB_182 isolation alone agsd Δ = −0.39%
    # SUB_183 NUMA alone agsd Δ = +1.54%
    print()
    print("## Superposition comparison")
    print("| lever | agsd Δ% |")
    print("|---|---:|")
    print("| SUB_182 (isolation alone) | −0.39% |")
    print("| SUB_183 (NUMA alone) | +1.54% |")
    print("| linear sum prediction | +1.15% |")
    if agg.get("agsd-gated"):
        print(f"| SUB_186 (stacked, measured) | {agg['agsd-gated']['delta_pct']:+.2f}% |")
        delta_meas = agg["agsd-gated"]["delta_pct"]
        delta_pred = -0.39 + 1.54  # +1.15%
        residual = delta_meas - delta_pred
        print(f"| residual (measured − predicted) | {residual:+.2f} pp |")
        if abs(residual) < 0.5:
            verdict = "linear_superposition_holds"
        elif residual > 0.5:
            verdict = "synergy (super-linear)"
        else:
            verdict = "interference / saturation (sub-linear)"
        print(f"| verdict | {verdict} |")

    out = {
        "per_cell": [
            {"mix": m, "scenario": sc, "off_tps": ot, "on_tps": nt, "delta_pct": d}
            for m, sc, ot, nt, d in rows
        ],
        "aggregate": agg,
    }
    (BASE / "aggregate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
