"""TSK_042 — per_request_raw.jsonl → oracle_table.parquet + 요약 (method spread, kill-gate 1차).

oracle_table 스키마(long): prompt_id, prompt_hash?, corpus, lang, n_input_tok,
  model, method, output_tps, wall_ms, completion_tokens, ok, seed.

사용: python build_oracle_table.py --raw per_request_raw.jsonl --out oracle_table.parquet
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = []
    with open(args.raw) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    ok = [r for r in recs if r.get("ok")]
    pq.write_table(pa.Table.from_pylist(recs), args.out)
    print(f"[table] {len(recs)} rows ({len(ok)} ok) → {args.out}")

    # (model, corpus, prompt_id) → {method: tps}  → method spread (kill-gate 1차)
    by_p = defaultdict(dict)
    for r in ok:
        by_p[(r["model"], r["corpus"], r["prompt_id"])][r["method"]] = r["output_tps"]
    spreads, by_corpus = [], defaultdict(list)
    for (model, corpus, _pid), m in by_p.items():
        if len(m) >= 2:
            vmax, vmin = max(m.values()), min(m.values())
            if vmax > 0:
                sp = (vmax - vmin) / vmax
                spreads.append(sp); by_corpus[(model, corpus)].append(sp)

    print("\n## method spread (prompt-level (max−min)/max) — kill-gate 1차")
    print("| model | corpus | n | mean spread | <5% 비율 |")
    print("|---|---|---:|---:|---:|")
    for (model, corpus), sps in sorted(by_corpus.items()):
        mean = sum(sps) / len(sps)
        small = sum(1 for s in sps if s < 0.05) / len(sps)
        print(f"| {model} | {corpus} | {len(sps)} | {mean*100:.1f}% | {small*100:.0f}% |")
    if spreads:
        overall = sum(spreads) / len(spreads)
        small = sum(1 for s in spreads if s < 0.05) / len(spreads)
        verdict = "KILL 위험 (method 차이 작음)" if overall < 0.05 else "OK (method 우열 존재)"
        print(f"\n→ overall mean spread {overall*100:.1f}% / <5% {small*100:.0f}% → **{verdict}**")
        print("  (method 차이가 충분해야 분류기/regret 이 의미. <5% 면 AGSD 가치 약함)")


if __name__ == "__main__":
    main()
