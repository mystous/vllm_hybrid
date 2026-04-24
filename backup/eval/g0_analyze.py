#!/usr/bin/env python3
"""G0 post-processing — parse HYBRID-CPU-PROFILE logs + bench json under
a sweep directory (measurement_results/<HW>/g0_<NN>/), compute batch
scaling, sublayer breakdown, and save plots + markdown summary.

Usage:
    eval/g0_analyze.py measurement_results/RTX3090/g0_00
    eval/g0_analyze.py measurement_results/H100x8/g0_05

Expected dir layout:
    <sweep_dir>/
    ├── seqs1/   hybrid_server_run.log, hybrid.json, applied_features.json
    ├── seqs2/
    ├── seqs4/
    ├── seqs8/
    └── seqs16/

Outputs (saved to <sweep_dir>/):
    analysis_summary.png        stacked bar + scaling curve
    analysis_sublayer_scaling.png  per-sublayer scaling
    analysis_bench.png          wall + TPOT
    analysis_summary.md         markdown table + findings
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r'\[HYBRID-CPU-PROFILE\] step=(\d+) reqs=(\S+) tokens=(\d+) '
    r'threads=(\d+) total=([\d.]+)ms '
    r'qkv=([\d.]+)ms\((\d+)\) o=([\d.]+)ms\((\d+)\) '
    r'gate_up=([\d.]+)ms\((\d+)\) down=([\d.]+)ms\((\d+)\) '
    r'norm=([\d.]+)ms\((\d+)\) act=([\d.]+)ms\((\d+)\) '
    r'attn_core=([\d.]+)ms\((\d+)\) other=([\d.]+)ms'
)

SEQS_CANDIDATES = [1, 2, 4, 8, 16, 32]


def parse_profile(log_path: Path) -> pd.DataFrame:
    rows = []
    if not log_path.exists():
        return pd.DataFrame()
    for line in log_path.read_text(errors='replace').splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        g = m.groups()
        rows.append({
            'step': int(g[0]), 'tokens': int(g[2]),
            'threads': int(g[3]), 'total_ms': float(g[4]),
            'qkv_ms': float(g[5]), 'o_ms': float(g[7]),
            'gate_up_ms': float(g[9]), 'down_ms': float(g[11]),
            'norm_ms': float(g[13]), 'act_ms': float(g[15]),
            'attn_core_ms': float(g[17]), 'other_ms': float(g[19]),
        })
    return pd.DataFrame(rows)


def parse_bench(run_dir: Path) -> dict:
    j = run_dir / 'hybrid.json'
    if not j.exists():
        return {}
    return json.loads(j.read_text())


def run(sweep_dir: Path) -> int:
    # Discover which seqs<N> dirs exist
    seqs = [s for s in SEQS_CANDIDATES if (sweep_dir / f'seqs{s}').exists()]
    if not seqs:
        print(f'ERROR: no seqs<N> subdirs found under {sweep_dir}', file=sys.stderr)
        return 2
    print(f'Sweep: {sweep_dir}')
    print(f'  seqs found: {seqs}')

    # Parse profile + bench
    dfs = {s: parse_profile(sweep_dir / f'seqs{s}' / 'hybrid_server_run.log') for s in seqs}
    for s in seqs:
        print(f'  seqs={s}: {len(dfs[s])} profile samples')

    # Per-seqs summary (median after warmup drop)
    summary = []
    for s in seqs:
        df = dfs[s]
        if df.empty:
            continue
        d = df.iloc[3:] if len(df) > 5 else df
        summary.append({
            'seqs': s, 'samples': len(d),
            'total_med': d['total_ms'].median(),
            'qkv_med': d['qkv_ms'].median(),
            'o_med': d['o_ms'].median(),
            'gate_up_med': d['gate_up_ms'].median(),
            'down_med': d['down_ms'].median(),
            'norm_med': d['norm_ms'].median(),
            'act_med': d['act_ms'].median(),
            'attn_core_med': d['attn_core_ms'].median(),
            'other_med': d['other_ms'].median(),
        })
    sdf = pd.DataFrame(summary).set_index('seqs')
    if sdf.empty:
        print('ERROR: no profile data parsed — check PROFILE=1 + sublayer flag', file=sys.stderr)
        return 3

    base_seq = sdf.index[0]
    base_total = sdf.loc[base_seq, 'total_med']
    sdf['scaling_ratio'] = sdf['total_med'] / base_total
    sdf['linear'] = sdf.index.values / base_seq
    sdf['efficiency'] = sdf['linear'] / sdf['scaling_ratio']
    print('\n=== Sublayer breakdown (median ms) ===')
    print(sdf.to_string())

    # Bench
    bench_rows = []
    for s in seqs:
        b = parse_bench(sweep_dir / f'seqs{s}')
        if not b:
            continue
        bench_rows.append({
            'seqs': s,
            'wall_s': b.get('duration'),
            'req_tps': b.get('request_throughput'),
            'out_tps': b.get('output_throughput'),
            'tpot_mean': b.get('mean_tpot_ms'),
            'tpot_p99': b.get('p99_tpot_ms'),
            'ttft_mean': b.get('mean_ttft_ms'),
            'ttft_p99': b.get('p99_ttft_ms'),
        })
    bdf = pd.DataFrame(bench_rows)
    if not bdf.empty:
        print('\n=== Bench summary ===')
        print(bdf.to_string(index=False))

    # Plot 1: stacked sublayer + scaling
    cols = ['qkv_med', 'o_med', 'gate_up_med', 'down_med',
            'norm_med', 'act_med', 'attn_core_med', 'other_med']
    labels = ['QKV', 'O proj', 'Gate+Up', 'Down', 'Norm', 'Act',
              'Attn core', 'Other']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']
    sub_cols = cols[:-1]
    sub_labels = labels[:-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    bottom = np.zeros(len(sdf))
    for c, lbl, clr in zip(cols, labels, colors):
        vals = sdf[c].values
        ax.bar(sdf.index.astype(str), vals, bottom=bottom, label=lbl,
               color=clr, edgecolor='white', linewidth=0.5)
        bottom += vals
    ax.set_xlabel('num_seqs (cpu_max_num_seqs)')
    ax.set_ylabel('Step time (ms, median)')
    ax.set_title(f'Sublayer breakdown  —  {sweep_dir.parent.name} / {sweep_dir.name}')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    xs = sdf.index.astype(float).values
    ax.plot(xs, sdf['scaling_ratio'].values, 'o-', color='#d62728',
            linewidth=2, markersize=10, label='Measured')
    ax.plot(xs, xs / xs[0], '--', color='gray', linewidth=1.5,
            label=f'Linear (N/{int(xs[0])})')
    ax.plot(xs, np.ones_like(xs), ':', color='green', linewidth=1.5,
            label='Perfect amortization')
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(s)) for s in xs])
    ax.set_xlabel('num_seqs')
    ax.set_ylabel(f'step(N) / step({int(xs[0])})')
    sr_max = sdf['scaling_ratio'].iloc[-1]
    ax.set_title(f'Batch scaling: ratio({int(xs[-1])}/1) = {sr_max:.2f}×  '
                 f'(linear {int(xs[-1])}×)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = sweep_dir / 'analysis_summary.png'
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    print(f'\nSaved: {out1}')

    # Plot 2: per-sublayer scaling
    fig, ax = plt.subplots(figsize=(11, 6))
    for c, lbl, clr in zip(sub_cols, sub_labels, colors):
        ratio = sdf[c] / sdf.loc[base_seq, c]
        ax.plot(xs, ratio.values, 'o-', label=lbl, color=clr)
    ax.plot(xs, xs / xs[0], '--', color='black', linewidth=1.5, label='Linear')
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(s)) for s in xs])
    ax.set_xlabel('num_seqs')
    ax.set_ylabel(f'sublayer time(N) / time({int(xs[0])})')
    ax.set_title('Per-sublayer batch scaling — top bottleneck 식별')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = sweep_dir / 'analysis_sublayer_scaling.png'
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    print(f'Saved: {out2}')

    # Plot 3: bench wall + TPOT
    if not bdf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.plot(bdf['seqs'], bdf['wall_s'], 'o-', color='#d62728',
                markersize=10, linewidth=2)
        ax.set_xscale('log', base=2)
        ax.set_xticks(bdf['seqs'])
        ax.set_xticklabels([str(int(s)) for s in bdf['seqs']])
        ax.set_xlabel('num_seqs')
        ax.set_ylabel('wall (s)')
        ax.set_title('Bench wall')
        ax.grid(True, alpha=0.3)
        for _, r in bdf.iterrows():
            ax.annotate(f"{r['wall_s']:.1f}s", (r['seqs'], r['wall_s']),
                        textcoords='offset points', xytext=(10, 5))
        ax = axes[1]
        ax.plot(bdf['seqs'], bdf['tpot_mean'], 'o-',
                label='TPOT mean', color='#1f77b4', markersize=10)
        ax.plot(bdf['seqs'], bdf['tpot_p99'], 's--',
                label='TPOT p99', color='#ff7f0e', markersize=8)
        ax.set_xscale('log', base=2)
        ax.set_xticks(bdf['seqs'])
        ax.set_xticklabels([str(int(s)) for s in bdf['seqs']])
        ax.set_xlabel('num_seqs')
        ax.set_ylabel('TPOT (ms)')
        ax.set_title('TPOT vs num_seqs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out3 = sweep_dir / 'analysis_bench.png'
        plt.savefig(out3, dpi=130, bbox_inches='tight')
        print(f'Saved: {out3}')

    # Top bottleneck
    base_row = sdf.loc[base_seq]
    parts = [(lbl, base_row[c], base_row[c] / base_row['total_med'] * 100)
             for c, lbl in zip(sub_cols + ['other_med'], sub_labels + ['Other'])]
    parts.sort(key=lambda x: -x[1])
    print(f'\n=== Top bottleneck @ seqs={base_seq} ===')
    for lbl, v, p in parts:
        bar = '█' * int(p / 2)
        print(f'  {lbl:12s} {v:6.2f}ms ({p:4.1f}%) {bar}')

    # Gate eval
    print('\n=== Gate evaluation ===')
    verdict = 'pre-G1'
    if 4 in sdf.index and 1 in sdf.index:
        r41 = sdf.loc[4, 'total_med'] / sdf.loc[1, 'total_med']
        print(f'scaling_ratio(4/1)  = {r41:.2f}×  (G1 ≤2.0×, G2 ≤1.5×)')
        if r41 <= 1.5:
            verdict = 'G2 pass'
        elif r41 <= 2.0:
            verdict = 'G1 pass'
        print(f'  -> {verdict}')
    if 16 in sdf.index and 1 in sdf.index:
        r161 = sdf.loc[16, 'total_med'] / sdf.loc[1, 'total_med']
        print(f'scaling_ratio(16/1) = {r161:.2f}× (linear 16×, failure factor {16/r161:.1f}×)')

    # Markdown summary
    md = sweep_dir / 'analysis_summary.md'
    try:
        tab = sdf[['total_med', 'qkv_med', 'o_med', 'gate_up_med', 'down_med',
                   'norm_med', 'act_med', 'attn_core_med', 'other_med',
                   'scaling_ratio']].to_markdown()
    except ImportError:
        tab = sdf[['total_med', 'gate_up_med', 'down_med', 'attn_core_med',
                   'scaling_ratio']].to_string()
    with open(md, 'w') as f:
        f.write(f'# G0 Analysis — {sweep_dir.parent.name} / {sweep_dir.name}\n\n')
        f.write(f'**Profile mode**: `VLLM_HYBRID_PROFILE=1 VLLM_HYBRID_PROFILE_SUBLAYER=1`\n\n')
        f.write('## Sublayer breakdown (median ms)\n\n')
        f.write(tab)
        f.write('\n\n')
        if not bdf.empty:
            try:
                btab = bdf.to_markdown(index=False)
            except ImportError:
                btab = bdf.to_string(index=False)
            f.write('## Bench metrics\n\n')
            f.write(btab)
            f.write('\n\n')
        f.write(f'## Verdict: **{verdict}**\n\n')
        f.write(f'Top bottleneck @ seqs={base_seq}:\n\n')
        for lbl, v, p in parts[:3]:
            f.write(f'1. {lbl}: {v:.2f}ms ({p:.1f}%)\n')
        f.write('\n## Plots\n\n')
        f.write('![Sublayer](./analysis_summary.png)\n\n')
        f.write('![Per-sublayer scaling](./analysis_sublayer_scaling.png)\n\n')
        if not bdf.empty:
            f.write('![Bench](./analysis_bench.png)\n')
    print(f'Saved: {md}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='G0 post-processing')
    ap.add_argument('sweep_dir', help='Path to measurement_results/<HW>/g0_<NN>/')
    args = ap.parse_args()
    sys.exit(run(Path(args.sweep_dir).resolve()))


if __name__ == '__main__':
    main()
