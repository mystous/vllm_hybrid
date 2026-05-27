"""SUB_183 aggregate: extract OFF/ON tps per (mix, scenario) and compute Δ%."""
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
    for sc in SCENARIOS:
        offs = [r[2] for r in rows if r[1] == sc and r[2] is not None]
        ons = [r[3] for r in rows if r[1] == sc and r[3] is not None]
        if len(offs) == 3 and len(ons) == 3:
            o = sum(offs) / 3
            n = sum(ons) / 3
            d = (n - o) / o * 100.0
            print(f"| {sc} | {o:,.1f} | {n:,.1f} | {d:+.2f}% |")
        else:
            print(f"| {sc} | (incomplete) | (incomplete) | — |")

    out = {
        "per_cell": [
            {"mix": m, "scenario": sc, "off_tps": ot, "on_tps": nt, "delta_pct": d}
            for m, sc, ot, nt, d in rows
        ],
    }
    (BASE / "aggregate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
