#!/usr/bin/env python3
"""CPU 112-core analysis: heatmap + NUMA placement + wait state breakdown.

Inputs (under <OUTDIR>/timeseries):
- per_core_util.jsonl     — 1Hz per-core util (224 logical core)
- per_thread_stat.jsonl   — per-thread CPU time + last_cpu + state
- thread_wchan.jsonl      — per-thread wait channel (where blocking)
- thread_affinity.jsonl   — Cpus_allowed_list (30 s 간격)
- numa_placement.jsonl    — NUMA page placement (30 s 간격)
- nvidia_smi.log

Outputs (under <OUTDIR>/analysis):
- heatmap_per_core.svg
- numa_placement.svg
- wait_state_breakdown.md
- numa_cross_socket.md
- summary.md
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def numa_node_of(core_id, node0, node1):
    return 0 if core_id in node0 else (1 if core_id in node1 else -1)


def heatmap_svg(samples, outpath, node0, node1):
    """N samples × 224 cores util heatmap."""
    if not samples:
        return
    n_samples = len(samples)
    n_cores = 224
    # color: green=idle, yellow=mid, red=busy
    def color(u):
        u = max(0, min(100, u))
        if u < 5:
            return "#fafafa"
        if u < 20:
            return f"#d4e9c4"
        if u < 40:
            return f"#b0d28a"
        if u < 60:
            return f"#f5d76e"
        if u < 80:
            return f"#f0934d"
        return f"#c64530"

    # downsample if too many samples
    max_w = 1400
    step = max(1, n_samples // max_w)
    samples = samples[::step]
    n_samples = len(samples)

    cell_w = max(1, min(2, max_w // max(1, n_samples)))
    cell_h = 3
    w = 200 + n_samples * cell_w
    h = 80 + n_cores * cell_h + 80

    parts = []
    parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Menlo, monospace" font-size="9">')
    parts.append(f'<rect width="{w}" height="{h}" fill="#fafafa"/>')
    parts.append(f'<text x="{w/2}" y="22" text-anchor="middle" font-size="13" font-weight="bold">112-core (224 HT) CPU utilization heatmap — 1 Hz × {n_samples * step} s sample</text>')
    parts.append(f'<text x="{w/2}" y="38" text-anchor="middle" font-size="10" fill="#555">left: NUMA node 0 (core 0-55 + 112-167, GPU 0-3), right: NUMA node 1 (core 56-111 + 168-223, GPU 4-7)</text>')

    # core 분류: physical 0-111 → HT pair 112-223
    # 표시 순서: 0-55 (n0 phys), 56-111 (n1 phys), 112-167 (n0 HT), 168-223 (n1 HT)
    core_order = list(range(0, 56)) + list(range(56, 112)) + list(range(112, 168)) + list(range(168, 224))

    y0 = 60
    for row, core in enumerate(core_order):
        y = y0 + row * cell_h
        node = numa_node_of(core, node0, node1)
        # row label every 8
        if row % 8 == 0:
            tag = f"c{core}"
            if core < 56:
                tag += " n0-phys"
            elif core < 112:
                tag += " n1-phys"
            elif core < 168:
                tag += " n0-HT"
            else:
                tag += " n1-HT"
            parts.append(f'<text x="195" y="{y + 8}" text-anchor="end" fill="#444">{tag}</text>')
        for col, samp in enumerate(samples):
            u = samp.get("cores", {}).get(str(core), {}).get("util_pct", 0)
            c = color(u)
            x = 200 + col * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{c}"/>')

    # node 0 / node 1 separator
    sep_y = y0 + 56 * cell_h
    parts.append(f'<line x1="0" y1="{sep_y}" x2="{w}" y2="{sep_y}" stroke="#0a3a6b" stroke-width="1.5" stroke-dasharray="6,3"/>')
    parts.append(f'<text x="195" y="{sep_y + 12}" text-anchor="end" font-size="9" fill="#0a3a6b" font-weight="bold">↓ NUMA 1 phys ↓</text>')
    sep_y2 = y0 + 112 * cell_h
    parts.append(f'<line x1="0" y1="{sep_y2}" x2="{w}" y2="{sep_y2}" stroke="#0a3a6b" stroke-width="1.5" stroke-dasharray="6,3"/>')
    parts.append(f'<text x="195" y="{sep_y2 + 12}" text-anchor="end" font-size="9" fill="#0a3a6b" font-weight="bold">↓ NUMA 0 HT ↓</text>')
    sep_y3 = y0 + 168 * cell_h
    parts.append(f'<line x1="0" y1="{sep_y3}" x2="{w}" y2="{sep_y3}" stroke="#0a3a6b" stroke-width="1.5" stroke-dasharray="6,3"/>')
    parts.append(f'<text x="195" y="{sep_y3 + 12}" text-anchor="end" font-size="9" fill="#0a3a6b" font-weight="bold">↓ NUMA 1 HT ↓</text>')

    # time axis
    axis_y = y0 + n_cores * cell_h + 12
    parts.append(f'<line x1="200" y1="{axis_y}" x2="{200 + n_samples * cell_w}" y2="{axis_y}" stroke="#444"/>')
    n_ticks = 10
    for i in range(n_ticks + 1):
        t_s = i * (n_samples * step) // n_ticks
        x = 200 + i * (n_samples * cell_w) // n_ticks
        parts.append(f'<line x1="{x}" y1="{axis_y - 3}" x2="{x}" y2="{axis_y + 3}" stroke="#444"/>')
        parts.append(f'<text x="{x}" y="{axis_y + 14}" text-anchor="middle" font-size="9">{t_s}s</text>')

    # legend
    parts.append(f'<g transform="translate(20, {h - 35})">')
    parts.append('<text x="0" y="0" font-size="10" font-weight="bold">util %:</text>')
    for i, (lo, c) in enumerate([(0, "#fafafa"), (5, "#d4e9c4"), (20, "#b0d28a"), (40, "#f5d76e"), (60, "#f0934d"), (80, "#c64530")]):
        x = 60 + i * 80
        parts.append(f'<rect x="{x}" y="-10" width="20" height="14" fill="{c}" stroke="#666"/>')
        parts.append(f'<text x="{x + 23}" y="0" font-size="9">≥{lo}</text>')
    parts.append('</g>')

    parts.append('</svg>')
    Path(outpath).write_text("\n".join(parts))


def numa_placement_md(samples, outpath, target_pids):
    """NUMA local vs remote 분석."""
    if not samples:
        Path(outpath).write_text("# NUMA placement\n\nNo data.\n")
        return
    last = samples[-1]
    lines = ["# NUMA placement (last sample)\n"]
    lines.append("| PID | NUMA 0 pages | NUMA 1 pages | local% (assumed) |")
    lines.append("|---|---:|---:|---:|")
    for pid, np_ in last.get("numa", {}).items():
        n0 = np_.get("0", 0)
        n1 = np_.get("1", 0)
        total = n0 + n1
        # assume worker 가 어느 NUMA 인지 status 에서 추정 못 하므로 단순 비율 표시
        lines.append(f"| {pid} | {n0:,} | {n1:,} | n0={100*n0/max(1,total):.1f}% n1={100*n1/max(1,total):.1f}% |")
    lines.append("")
    lines.append("## status snapshots (last sample)")
    lines.append("")
    for pid, st in last.get("status", {}).items():
        lines.append(f"### PID {pid}")
        for k, v in st.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    Path(outpath).write_text("\n".join(lines))


def wait_state_md(samples, outpath):
    """wchan 분포 분석."""
    if not samples:
        Path(outpath).write_text("# Wait state\n\nNo data.\n")
        return
    wchan_count = Counter()
    pid_wchan = defaultdict(Counter)
    for s in samples:
        for key, w in s.get("wchan", {}).items():
            wchan_count[w] += 1
            pid = key.split("/")[0]
            pid_wchan[pid][w] += 1
    total = sum(wchan_count.values())
    lines = ["# Wait state breakdown (per-thread wchan, all samples)\n"]
    lines.append(f"Total wait events: {total}\n")
    lines.append("## Top 20 wait channels (overall)\n")
    lines.append("| wchan | count | % |")
    lines.append("|---|---:|---:|")
    for w, c in wchan_count.most_common(20):
        lines.append(f"| `{w}` | {c} | {100*c/max(1,total):.1f}% |")
    lines.append("")
    lines.append("## Per-PID top wait channels (top 5)\n")
    for pid in sorted(pid_wchan):
        ptotal = sum(pid_wchan[pid].values())
        lines.append(f"### PID {pid}  (total wait events = {ptotal})")
        lines.append("| wchan | count | % |")
        lines.append("|---|---:|---:|")
        for w, c in pid_wchan[pid].most_common(5):
            lines.append(f"| `{w}` | {c} | {100*c/max(1,ptotal):.1f}% |")
        lines.append("")
    Path(outpath).write_text("\n".join(lines))


def thread_cpu_breakdown(samples, outpath, node0, node1):
    """per-thread CPU 누적 + last_cpu 분포 → NUMA cross-socket 검증."""
    if not samples:
        Path(outpath).write_text("# Thread CPU\n\nNo data.\n")
        return
    last_cpu_count = defaultdict(Counter)  # (pid, comm) → Counter(last_cpu)
    cpu_total = defaultdict(int)  # (pid, comm) → cumulative jiffies
    for s in samples:
        for key, t in s.get("threads", {}).items():
            pid, tid = key.split("/")
            comm = t.get("comm", "?")
            lc = t.get("last_cpu", -1)
            if lc >= 0:
                last_cpu_count[(pid, comm)][lc] += 1
            cpu_total[(pid, comm)] += t.get("d_utime", 0) + t.get("d_stime", 0)
    lines = ["# Per-thread CPU breakdown — last_cpu 분포\n"]
    lines.append("PID = worker process. comm = thread name. last_cpu_count = 그 thread 가 본 sample 마다 어느 CPU 에서 running 이었나.\n")
    # 누적 jiffies top 30
    lines.append("## Top 30 (pid, comm) by cumulative CPU time\n")
    lines.append("| pid | comm | cumulative jiff | last_cpu top-5 (NUMA 분포) |")
    lines.append("|---|---|---:|---|")
    for (pid, comm), tot in sorted(cpu_total.items(), key=lambda x: -x[1])[:30]:
        tops = last_cpu_count[(pid, comm)].most_common(5)
        # NUMA node 분포
        n0_cnt = sum(c for cpu, c in last_cpu_count[(pid, comm)].items() if cpu in node0)
        n1_cnt = sum(c for cpu, c in last_cpu_count[(pid, comm)].items() if cpu in node1)
        tot_cnt = n0_cnt + n1_cnt
        if tot_cnt:
            numa_str = f"n0={100*n0_cnt/tot_cnt:.0f}% n1={100*n1_cnt/tot_cnt:.0f}%"
        else:
            numa_str = "?"
        topstr = ", ".join([f"c{cpu}×{c}" for cpu, c in tops])
        lines.append(f"| {pid} | `{comm}` | {tot:,} | [{numa_str}] {topstr} |")
    Path(outpath).write_text("\n".join(lines))


def parse_node_cpulist(path):
    n0, n1 = set(), set()
    if not Path(path).exists():
        # fallback: hardcoded from numactl
        n0 = set(range(0, 56)) | set(range(112, 168))
        n1 = set(range(56, 112)) | set(range(168, 224))
        return n0, n1
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("node0"):
                rng = line.split(":", 1)[1].strip()
            elif line.startswith("node1"):
                rng = line.split(":", 1)[1].strip()
            else:
                continue
            for tok in rng.split(","):
                tok = tok.strip()
                if "-" in tok:
                    a, b = tok.split("-")
                    rng_set = set(range(int(a), int(b) + 1))
                else:
                    rng_set = {int(tok)}
                if line.startswith("node0"):
                    n0 |= rng_set
                else:
                    n1 |= rng_set
    return n0, n1


def summary_md(outdir, core_samples, thread_samples, wchan_samples, numa_samples, n0, n1):
    """전체 분석 summary + heatmap aggregate."""
    lines = ["# CPU 112-core Analysis Summary\n"]
    if not core_samples:
        lines.append("No core sample data.")
        Path(outdir / "summary.md").write_text("\n".join(lines))
        return
    # 평균 util per core, NUMA node
    n_samp = len(core_samples)
    sum_util = defaultdict(float)
    for s in core_samples:
        for cid, c in s.get("cores", {}).items():
            sum_util[int(cid)] += c.get("util_pct", 0)
    avg_util = {cid: u / n_samp for cid, u in sum_util.items()}

    # NUMA aggregate
    n0_util = [avg_util.get(c, 0) for c in n0]
    n1_util = [avg_util.get(c, 0) for c in n1]
    lines.append(f"## Run fact")
    lines.append(f"- Sample count: {n_samp} (1 Hz)")
    lines.append(f"- 224 logical core total")
    lines.append(f"- NUMA 0: {len(n0)} core, avg util = {sum(n0_util)/len(n0_util):.1f}%")
    lines.append(f"- NUMA 1: {len(n1)} core, avg util = {sum(n1_util)/len(n1_util):.1f}%")
    lines.append(f"- system-wide avg util = {sum(avg_util.values())/len(avg_util):.1f}%")
    lines.append("")
    # Top busy core
    lines.append("## Top 20 busiest cores (by avg util)\n")
    lines.append("| core | NUMA | avg util % | physical/HT |")
    lines.append("|---|---|---:|---|")
    sorted_cores = sorted(avg_util.items(), key=lambda x: -x[1])[:20]
    for cid, u in sorted_cores:
        node = "n0" if cid in n0 else ("n1" if cid in n1 else "?")
        phys = "phys" if cid < 112 else "HT"
        lines.append(f"| c{cid} | {node} | {u:.1f} | {phys} |")
    lines.append("")
    lines.append("## Bottom 20 (least used)\n")
    lines.append("| core | NUMA | avg util % | physical/HT |")
    lines.append("|---|---|---:|---|")
    for cid, u in sorted(avg_util.items(), key=lambda x: x[1])[:20]:
        node = "n0" if cid in n0 else ("n1" if cid in n1 else "?")
        phys = "phys" if cid < 112 else "HT"
        lines.append(f"| c{cid} | {node} | {u:.1f} | {phys} |")
    lines.append("")
    # util distribution histogram
    lines.append("## Util distribution histogram\n")
    bins = [(0, 5), (5, 20), (20, 40), (40, 60), (60, 80), (80, 100.1)]
    for lo, hi in bins:
        count = sum(1 for u in avg_util.values() if lo <= u < hi)
        lines.append(f"- {lo:.0f}-{hi:.0f}%: {count} cores")
    lines.append("")
    Path(outdir / "summary.md").write_text("\n".join(lines))


def main():
    if len(sys.argv) < 2:
        print("usage: cpu112_analyze.py <OUTDIR>", file=sys.stderr)
        sys.exit(1)
    outdir = Path(sys.argv[1])
    ts = outdir / "timeseries"
    ana = outdir / "analysis"
    ana.mkdir(exist_ok=True)
    env = outdir / "env"
    n0, n1 = parse_node_cpulist(env / "node_cpulist.txt")
    print(f"[analyze] node0 has {len(n0)} cores, node1 has {len(n1)} cores")

    print("[analyze] loading per_core_util.jsonl...")
    core_samples = load_jsonl(ts / "per_core_util.jsonl")
    print(f"  {len(core_samples)} samples")

    print("[analyze] loading per_thread_stat.jsonl...")
    thread_samples = load_jsonl(ts / "per_thread_stat.jsonl")
    print(f"  {len(thread_samples)} samples")

    print("[analyze] loading thread_wchan.jsonl...")
    wchan_samples = load_jsonl(ts / "thread_wchan.jsonl")
    print(f"  {len(wchan_samples)} samples")

    print("[analyze] loading numa_placement.jsonl...")
    numa_samples = load_jsonl(ts / "numa_placement.jsonl")
    print(f"  {len(numa_samples)} samples")

    print("[analyze] generating heatmap...")
    heatmap_svg(core_samples, ana / "heatmap_per_core.svg", n0, n1)

    print("[analyze] numa placement md...")
    target_pids = []
    pids_file = env / "target_pids.txt"
    if pids_file.exists():
        target_pids = [int(x) for x in pids_file.read_text().strip().split(",") if x.strip()]
    numa_placement_md(numa_samples, ana / "numa_placement.md", target_pids)

    print("[analyze] wait state md...")
    wait_state_md(wchan_samples, ana / "wait_state_breakdown.md")

    print("[analyze] thread breakdown md...")
    thread_cpu_breakdown(thread_samples, ana / "thread_cpu_breakdown.md", n0, n1)

    print("[analyze] summary md...")
    summary_md(ana, core_samples, thread_samples, wchan_samples, numa_samples, n0, n1)

    print(f"[analyze] done. outputs in {ana}")


if __name__ == "__main__":
    main()
