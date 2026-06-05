# SPDX-License-Identifier: Apache-2.0
"""Phase A4-exclusive: cross-mode 10-case byte-equal verifier (v3).

Reads the per-mode `llama8b_<MODE>_v3.sample10.jsonl` files and reports
SHA256 / byte-length matches per prompt-id across modes.

Usage:
    python compare_sha_v3.py            # compares baseline / double / exclusive
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

POC = Path(__file__).resolve().parent
MODES = ["baseline", "double", "exclusive"]


def load(mode: str) -> dict[int, dict]:
    f = POC / f"llama8b_{mode}_v3.sample10.jsonl"
    if not f.exists():
        return {}
    by_idx: dict[int, dict] = {}
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_idx[r["idx"]] = r
    return by_idx


def main() -> int:
    data = {m: load(m) for m in MODES}
    missing = [m for m, d in data.items() if not d]
    if missing:
        print(f"MISSING sample10 file(s): {missing}", file=sys.stderr)
        return 2
    idxs = sorted(data[MODES[0]].keys())
    header = ["idx"] + [f"{m}_bytes" for m in MODES] + [f"{m}_sha[:16]" for m in MODES] + [
        "base_eq_double", "base_eq_excl"
    ]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    n_match_double = 0
    n_match_excl = 0
    for idx in idxs:
        rows = {m: data[m].get(idx) for m in MODES}
        if any(r is None or not r.get("ok") for r in rows.values()):
            print(f"{idx}: skipped (missing or not ok in some mode)")
            continue
        b = {m: rows[m]["text_bytes_len"] for m in MODES}
        s = {m: rows[m]["text_sha"][:16] for m in MODES}
        base_eq_double = (s["baseline"] == s["double"]) and (b["baseline"] == b["double"])
        base_eq_excl = (s["baseline"] == s["exclusive"]) and (b["baseline"] == b["exclusive"])
        if base_eq_double:
            n_match_double += 1
        if base_eq_excl:
            n_match_excl += 1
        row = [str(idx)] + [str(b[m]) for m in MODES] + [s[m] for m in MODES] + [
            "OK" if base_eq_double else "MISMATCH",
            "OK" if base_eq_excl else "MISMATCH",
        ]
        print(" | ".join(row))
    print()
    print(f"baseline ≡ double    : {n_match_double}/{len(idxs)}")
    print(f"baseline ≡ exclusive : {n_match_excl}/{len(idxs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
