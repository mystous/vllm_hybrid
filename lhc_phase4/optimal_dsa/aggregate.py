#!/usr/bin/env python3
"""Aggregate Optimal+DSA real-corpus sweep across all models.

vanilla/suffix baselines  ← TSK_042 (vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602)
dsa/suffix_dsa cells       ← fresh runs (lhc_phase4/optimal_dsa/runs)

Emits MEASUREMENTS.md with 4-config × 7-corpus matrix per model + cross-model
verdict on whether DSA stacks on top of suffix.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
TSK042 = Path("/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/"
              "runs/tput_t1t3_20260602")

CONFIGS = ["vanilla", "dsa", "suffix", "suffix_dsa"]
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
]


def load_cell(tag, cfg, corpus):
    """vanilla/suffix → TSK_042; dsa/suffix_dsa → our fresh runs."""
    if cfg in ("vanilla", "suffix"):
        p = TSK042 / f"summ_{tag}_{cfg}_{corpus}.json"
    else:
        p = RUNS / f"summ_{tag}_{cfg}_{corpus}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt_tps(d):
    return "—" if d is None else f"{d.get('output_tps', 0):,.0f}"


def fmt_delta(base, cur):
    if base is None or cur is None:
        return "—"
    b = base.get("output_tps")
    c = cur.get("output_tps")
    if not b:
        return "—"
    d = 100.0 * (c - b) / b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


def cells_present(tag):
    """Count fresh dsa+suffix_dsa cells (0-14)."""
    n = 0
    for cfg in ("dsa", "suffix_dsa"):
        for c in CORPORA:
            if (RUNS / f"summ_{tag}_{cfg}_{c}.json").exists():
                n += 1
    return n


def main():
    out = []
    out.append("# Optimal+DSA Multi-Model Validation — TSK_042 baseline\n\n")
    out.append("**HW**: DGX B200 × 8 (sm_100), Xeon Platinum 8570 + AMX + DSA 8 SWQ\n")
    out.append("**Harness**: `realistic_eval/throughput_runner.py` (TSK_042 schema)\n")
    out.append("**Setup**: corpus 500p × conc=32 × max_tokens=8192, streaming, "
               "`cudagraph_mode=FULL_AND_PIECEWISE` (FaP)\n")
    out.append("**Baseline source**:\n")
    out.append(f"- `vanilla`, `suffix` ← TSK_042 ({TSK042.name})\n")
    out.append(f"- `dsa`, `suffix_dsa` ← fresh ({RUNS.name})\n\n")
    out.append("**Configs**: vanilla / dsa / suffix(=Optimal) / suffix_dsa(=Optimal+DSA), n=1\n\n")

    # Coverage
    out.append("## Coverage (fresh dsa + suffix_dsa cells)\n\n")
    out.append("| model | cells (out of 14) |\n|---|---:|\n")
    for tag in MODELS:
        n = cells_present(tag)
        mark = "✅" if n == 14 else ("🟡" if n > 0 else "⚪")
        out.append(f"| {tag} | {mark} {n}/14 |\n")
    out.append("\n")

    # Headline — mix corpus
    out.append("## Headline — mix corpus (모든 모델)\n\n")
    out.append("| model | vanilla | dsa | suffix (Optimal) | **suffix_dsa (Opt+DSA)** | "
               "DSA vs van | suffix vs van | **suf_dsa vs van** | **DSA gain on suffix** |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag in MODELS:
        v = load_cell(tag, "vanilla", "mix")
        d = load_cell(tag, "dsa", "mix")
        s = load_cell(tag, "suffix", "mix")
        sd = load_cell(tag, "suffix_dsa", "mix")
        if all(x is None for x in (v, d, s, sd)):
            continue
        row = [
            tag,
            fmt_tps(v), fmt_tps(d), fmt_tps(s), f"**{fmt_tps(sd)}**",
            fmt_delta(v, d), fmt_delta(v, s),
            f"**{fmt_delta(v, sd)}**", f"**{fmt_delta(s, sd)}**",
        ]
        out.append("| " + " | ".join(row) + " |\n")
    out.append("\n")

    # Per-model detailed tables
    for tag in MODELS:
        n = cells_present(tag)
        if n == 0:
            continue
        out.append(f"---\n\n## {tag} — output_tps (4 config × 7 corpus)\n\n")
        out.append("| corpus | vanilla | dsa | suffix | **suffix_dsa** | "
                   "DSA Δ | suffix Δ | **suf_dsa Δ** | stack Δ (vs suf) |\n")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for c in CORPORA:
            v = load_cell(tag, "vanilla", c)
            d = load_cell(tag, "dsa", c)
            s = load_cell(tag, "suffix", c)
            sd = load_cell(tag, "suffix_dsa", c)
            row = [
                c,
                fmt_tps(v), fmt_tps(d), fmt_tps(s), f"**{fmt_tps(sd)}**",
                fmt_delta(v, d), fmt_delta(v, s),
                f"**{fmt_delta(v, sd)}**", fmt_delta(s, sd),
            ]
            out.append("| " + " | ".join(row) + " |\n")
        out.append("\n")

    # Verdict — DSA stack effect per (model, corpus)
    out.append("---\n\n## Verdict — DSA가 suffix 위에 가산되는가? (Δ stack = suffix_dsa vs suffix)\n\n")
    out.append("| model | sharegpt | swebench | humaneval | mbpp | wildchat | lmsys | mix | **avg** |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag in MODELS:
        row = [tag]
        deltas = []
        for c in CORPORA:
            s = load_cell(tag, "suffix", c)
            sd = load_cell(tag, "suffix_dsa", c)
            if s is None or sd is None or not s.get("output_tps"):
                row.append("—")
            else:
                dd = 100.0 * (sd["output_tps"] - s["output_tps"]) / s["output_tps"]
                deltas.append(dd)
                sign = "+" if dd >= 0 else ""
                row.append(f"{sign}{dd:.1f}%")
        if deltas:
            avg = sum(deltas) / len(deltas)
            sign = "+" if avg >= 0 else ""
            row.append(f"**{sign}{avg:.2f}%**")
        else:
            row.append("—")
        out.append("| " + " | ".join(row) + " |\n")

    text = "".join(out)
    (ROOT / "MEASUREMENTS.md").write_text(text)
    print(text[-3000:])
    print(f"\n[wrote] {ROOT / 'MEASUREMENTS.md'} ({len(text)} chars)")


if __name__ == "__main__":
    main()
