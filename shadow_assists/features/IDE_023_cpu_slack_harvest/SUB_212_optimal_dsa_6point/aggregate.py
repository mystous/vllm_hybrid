#!/usr/bin/env python3
"""Aggregate 6-point Optimal+DSA sweep into unified MEASUREMENTS.md.

6 points per (model, corpus):
  ① van(OFF)   = TSK_042 vanilla (host DSA WQ disabled, 2026-06-02)
  ② van(ON)    = fresh vanilla   (host DSA WQ enabled,  2026-06-10+)
  ③ DSA(ON)    = vllm VLLM_LHC_DSA=1 + VLLM_LEVER_N9=1 on top of host DSA ON
  ④ suf(OFF)   = TSK_042 suffix (host DSA WQ disabled)
  ⑤ suf(ON)    = fresh suffix    (host DSA WQ enabled)
  ⑥ suf+dsa(ON)= fresh suffix + vllm DSA env on (host DSA ON)

Covers 9 models (8 standard + R1-671B) full 6/6, plus Llama-405B-FP8 4/6
(⑤⑥ engine init failure with suffix+FP8+B200 single instance).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
TSK042 = Path("/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/"
              "runs/tput_t1t3_20260602")
CORPORA = ["sharegpt", "swebench", "humaneval", "mbpp", "wildchat", "lmsys", "mix"]
MODELS = [
    "Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-32B-Instruct",
    "DeepSeek-R1-Distill-Qwen-32B",
    "Qwen2.5-72B-Instruct",
    "Llama-3.1-70B-Instruct",
    "DeepSeek-R1-Distill-Llama-70B",
    "Llama-3.1-405B-Instruct-FP8",
    "DeepSeek-R1",
]
# 6-point mapping: (label, base_dir, config_string)
POINTS = [
    ("van(OFF)",   TSK042, "vanilla"),
    ("van(ON)",    RUNS,   "vanilla"),
    ("DSA(ON)",    RUNS,   "dsa"),
    ("suf(OFF)",   TSK042, "suffix"),
    ("suf(ON)",    RUNS,   "suffix"),
    ("suf+dsa(ON)",RUNS,   "suffix_dsa"),
]


def load_cell(tag, base, cfg, corpus):
    p = base / f"summ_{tag}_{cfg}_{corpus}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def get_tps(d):
    return d.get("output_tps") if d else None


def fmt_tps(v):
    return f"{v:,.0f}" if v is not None else "—"


def fmt_delta(base, cur):
    if base is None or cur is None or base == 0:
        return "—"
    d = 100.0 * (cur - base) / base
    s = "+" if d >= 0 else ""
    return f"{s}{d:.1f}%"


def main():
    out = []
    out.append("# Optimal+DSA 6-Point Coverage — Multi-Model Real-Corpus Validation\n\n")
    out.append("**HW**: DGX B200 × 8 (sm_100), Xeon Platinum 8570 + AMX + DSA 8 SWQ\n")
    out.append("**Harness**: `vllm_config_perf/gating/realistic_eval/throughput_runner.py`\n")
    out.append("**Setup**: corpus 500p × conc=32 × max_tok=8192, streaming, "
               "`cudagraph_mode=FULL_AND_PIECEWISE` (FaP)\n\n")
    out.append("## 6 measurement points per (model, corpus)\n\n")
    out.append("| ID | label | host DSA WQ | vllm spec decode | vllm DSA env | source |\n")
    out.append("|---|---|:---:|:---:|:---:|---|\n")
    out.append("| ① | van(OFF) | disabled | none | none | TSK_042 (2026-06-02) |\n")
    out.append("| ② | van(ON) | **enabled** | none | none | fresh (2026-06-10+) |\n")
    out.append("| ③ | DSA(ON) | enabled | none | **on** | fresh |\n")
    out.append("| ④ | suf(OFF) | disabled | suffix K=32 | none | TSK_042 |\n")
    out.append("| ⑤ | suf(ON) | **enabled** | suffix K=32 | none | fresh |\n")
    out.append("| ⑥ | suf+dsa(ON) | enabled | suffix K=32 | **on** | fresh |\n\n")

    # Coverage summary
    out.append("---\n\n## Coverage\n\n")
    out.append("| model | ① | ② | ③ | ④ | ⑤ | ⑥ | 셀 합 |\n")
    out.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n")
    grand = 0
    for tag in MODELS:
        counts = []
        for label, base, cfg in POINTS:
            n = sum(1 for c in CORPORA if (base / f"summ_{tag}_{cfg}_{c}.json").exists())
            counts.append("✅" if n == 7 else f"{n}/7")
            grand += n
        out.append(f"| `{tag}` | " + " | ".join(counts) + f" | **{sum(POINTS.__class__([n if isinstance(n,int) else int(n.split('/')[0]) if isinstance(n,str) and '/' in n else 7 for n in counts]))}/42** |\n")
    # Replace problematic line (simpler counting)
    out_lines = out
    out = out[:-len(MODELS)]
    for tag in MODELS:
        counts_n = []
        cells = []
        for label, base, cfg in POINTS:
            n = sum(1 for c in CORPORA if (base / f"summ_{tag}_{cfg}_{c}.json").exists())
            counts_n.append(n)
            cells.append("✅" if n == 7 else f"{n}/7")
        total = sum(counts_n)
        out.append(f"| `{tag}` | " + " | ".join(cells) + f" | **{total}/42** |\n")
    out.append(f"\n**전체: {grand}/420 = {grand/420*100:.1f}%**\n\n")

    # Main 6-point table — mix corpus (most representative)
    out.append("---\n\n## Headline — mix corpus (10 모델 × 6 points)\n\n")
    out.append("| model | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) | **best** |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|:---:|\n")
    for tag in MODELS:
        vals = {}
        for label, base, cfg in POINTS:
            d = load_cell(tag, base, cfg, "mix")
            vals[label] = get_tps(d)
        nonempty = [(l, v) for l, v in vals.items() if v is not None]
        best = max(nonempty, key=lambda x: x[1]) if nonempty else None
        row = [f"`{tag}`"] + [fmt_tps(vals[l]) for l, _, _ in POINTS]
        if best:
            row.append(f"**{best[0]} {best[1]:,.0f}**")
        else:
            row.append("—")
        out.append("| " + " | ".join(row) + " |\n")

    # Per-model 6-point × 7 corpus tables
    for tag in MODELS:
        out.append(f"\n---\n\n## `{tag}` — 6 points × 7 corpus\n\n")
        out.append("| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |\n")
        out.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for c in CORPORA:
            row = [c]
            for label, base, cfg in POINTS:
                d = load_cell(tag, base, cfg, c)
                row.append(fmt_tps(get_tps(d)))
            out.append("| " + " | ".join(row) + " |\n")

    # Effect decomposition
    out.append("\n---\n\n## Effect decomposition — mix corpus (Δ% 분해)\n\n")
    out.append("- **host DSA effect on vanilla**: ② vs ①\n")
    out.append("- **host DSA effect on suffix**: ⑤ vs ④\n")
    out.append("- **vllm env effect on vanilla (host ON)**: ③ vs ②\n")
    out.append("- **vllm env effect on suffix (host ON)**: ⑥ vs ⑤\n")
    out.append("- **suffix effect (same-state host OFF)**: ④ vs ①\n")
    out.append("- **suffix effect (same-state host ON)**: ⑤ vs ②\n\n")
    out.append("| model | DSA on van | DSA on suf | vllm env on van | vllm env on suf | suf-gain (OFF) | suf-gain (ON) |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for tag in MODELS:
        vals = {}
        for label, base, cfg in POINTS:
            d = load_cell(tag, base, cfg, "mix")
            vals[label] = get_tps(d)
        row = [f"`{tag}`",
               fmt_delta(vals["van(OFF)"], vals["van(ON)"]),
               fmt_delta(vals["suf(OFF)"], vals["suf(ON)"]),
               fmt_delta(vals["van(ON)"],  vals["DSA(ON)"]),
               fmt_delta(vals["suf(ON)"],  vals["suf+dsa(ON)"]),
               fmt_delta(vals["van(OFF)"], vals["suf(OFF)"]),
               fmt_delta(vals["van(ON)"],  vals["suf(ON)"])]
        out.append("| " + " | ".join(row) + " |\n")

    # Winner distribution
    out.append("\n---\n\n## Winner distribution — best point per (model, corpus) 70 셀\n\n")
    counts = {label: 0 for label, _, _ in POINTS}
    counts["—"] = 0
    for tag in MODELS:
        for c in CORPORA:
            vals = {}
            for label, base, cfg in POINTS:
                d = load_cell(tag, base, cfg, c)
                vals[label] = get_tps(d)
            nonempty = [(l, v) for l, v in vals.items() if v is not None]
            if nonempty:
                best = max(nonempty, key=lambda x: x[1])
                counts[best[0]] += 1
            else:
                counts["—"] += 1
    total_cells = sum(counts.values())
    out.append("| Winner point | count | % |\n")
    out.append("|---|---:|---:|\n")
    for label, _, _ in POINTS:
        n = counts[label]
        out.append(f"| {label} | **{n}** | {n/total_cells*100:.1f}% |\n")
    if counts["—"] > 0:
        out.append(f"| — (missing) | {counts['—']} | {counts['—']/total_cells*100:.1f}% |\n")

    text = "".join(out)
    out_path = ROOT / "MEASUREMENTS_6point.md"
    out_path.write_text(text)
    print(f"[wrote] {out_path}")
    print(f"  total cells: {grand}/420 = {grand/420*100:.1f}%")
    print(f"  doc size: {len(text)} chars")


if __name__ == "__main__":
    main()
