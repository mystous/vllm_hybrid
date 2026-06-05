"""Diff two correctness_check.py outputs (JF off vs JF on).

Computes:
  - parse-ok rate per side
  - schema-conform rate per side
  - byte-equal rate (informational)
  - per-sample diffs for the first few mismatches
"""
from __future__ import annotations

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="JSON from JF-off run")
    ap.add_argument("--on", required=True, help="JSON from JF-on run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    off = json.load(open(args.off))
    on_ = json.load(open(args.on))
    assert off["n"] == on_["n"]
    n = off["n"]

    parse_off = off["parse_ok_count"]
    parse_on = on_["parse_ok_count"]
    conf_off = off["conform_ok_count"]
    conf_on = on_["conform_ok_count"]

    # By-sample byte-equality + JSON-equivalence
    byte_eq = 0
    json_eq = 0
    diffs = []
    for a, b in zip(off["samples"], on_["samples"]):
        if not a.get("ok") or not b.get("ok"):
            diffs.append({"i": a.get("i"), "kind": "error",
                          "off_err": a.get("error"), "on_err": b.get("error")})
            continue
        # Compare the extracted JSON object (the schema-constrained span),
        # not the trailing free-form text the model may continue to emit.
        ta = a.get("extracted_json") or a["raw_text"]
        tb = b.get("extracted_json") or b["raw_text"]
        if ta == tb:
            byte_eq += 1
            json_eq += 1
            continue
        try:
            ja = json.loads(ta) if a.get("parse_ok") else None
            jb = json.loads(tb) if b.get("parse_ok") else None
            if ja == jb and ja is not None:
                json_eq += 1
        except Exception:
            pass
        if len(diffs) < 5:
            diffs.append({"i": a["i"], "kind": "diff",
                          "off_text": ta[:200], "on_text": tb[:200]})

    out = {
        "n": n,
        "parse_ok": {"off": parse_off, "on": parse_on},
        "schema_conform": {"off": conf_off, "on": conf_on},
        "byte_equal_count": byte_eq,
        "json_equivalent_count": json_eq,
        "first_diffs": diffs,
    }
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
