"""SUB_201/L4 — 결과 summary table 생성.

읽기: qwen7b_{A_vanilla,B_ngram,C_ngram_glb}.json
출력: table (tps, TTFT p50, TPOT p50, GPU%, CPU%, accept_rate, n_ok, wall, Δ% vs A)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

POC = Path(__file__).parent
MODES = ["A_vanilla", "B_ngram", "C_ngram_glb"]

rows = []
for m in MODES:
    p = POC / f"qwen7b_{m}.json"
    if not p.exists():
        print(f"[skip] {p.name} missing")
        continue
    d = json.loads(p.read_text())
    rows.append((m, d))

if not rows:
    print("no results found")
    sys.exit(0)

base_tps = None
for m, d in rows:
    if m == "A_vanilla":
        base_tps = d["output_tps"]
        break

print(f"\n{'mode':<14} {'tps':>8} {'Δ%':>7} {'TTFT p50':>10} {'TPOT p50':>10} {'gpu%':>6} {'cpu%':>6} {'α':>7} {'acc/draft':>14} {'n_ok':>6}")
print("-" * 100)
for m, d in rows:
    tps = d.get("output_tps") or 0
    delta = (tps - base_tps) / base_tps * 100 if base_tps else 0.0
    ttft = d.get("ttft_ms_p50")
    tpot = d.get("tpot_ms_p50")
    gpu = d.get("gpu_util")
    cpu = d.get("cpu_util")
    alpha = d.get("accept_rate")
    acc = d.get("accept_tokens")
    draft = d.get("draft_tokens")
    n_ok = d.get("n_ok")
    accs = f"{int(acc) if acc else '-'}/{int(draft) if draft else '-'}" if acc is not None else "-/-"
    print(f"{m:<14} {tps:>8.1f} {delta:>+6.1f}% {str(ttft):>10} {str(tpot):>10} {str(gpu):>6} {str(cpu):>6} {str(alpha):>7} {accs:>14} {str(n_ok):>6}")
