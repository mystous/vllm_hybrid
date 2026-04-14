#!/usr/bin/env python3
"""
inspect.py — Single experiment hardware utilization inspector

Reports:
  - CPU pinning: which CPUs were pinned per engine, physical vs HT
  - Per-NUMA CPU utilization: physical primary / HT sibling breakdown
  - GPU utilization: per-card and average

Usage:
    python inspect.py <result_dir>
    python inspect.py eval/basic/H100x8/20260414_054010_*

Output:
    Console + <result_dir>/inspect.txt
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# System info parsing
# ---------------------------------------------------------------------------

def load_system_info(run_dir: Path) -> dict:
    p = run_dir / "system_info.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def build_cpu_maps(si: dict) -> tuple[dict, int, int, int]:
    """
    Returns:
        cpu_to_numa  : {cpu_id: numa_node}
        total_cpus   : int
        tpc          : threads_per_core
        phys_thresh  : CPU id < phys_thresh → physical primary
    """
    cpu_to_numa: dict[int, int] = {}
    numa_raw = si.get("numa", {}).get("raw", "")
    for line in numa_raw.splitlines():
        m = re.match(r"node (\d+) cpus: (.+)", line)
        if m:
            node = int(m.group(1))
            for c in m.group(2).split():
                cpu_to_numa[int(c)] = node

    total_cpus = si.get("cpu", {}).get("total_cpus", 0)
    tpc        = si.get("cpu", {}).get("threads_per_core", 1)
    phys_thresh = total_cpus // tpc  # CPU < phys_thresh → physical primary

    return cpu_to_numa, total_cpus, tpc, phys_thresh


def build_column_map(num_cols: int, tpc: int,
                     cpu_to_numa: dict, phys_thresh: int) -> list[dict]:
    """
    column_map[col_idx] = {
        "cpu_ids":  [cpu_a, cpu_b, ...],
        "numa":     int,
        "is_phys":  bool,
    }
    """
    col_map = []
    for i in range(num_cols):
        base = i * tpc
        cpu_ids = list(range(base, base + tpc))
        numa = cpu_to_numa.get(base, -1)
        is_phys = base < phys_thresh
        col_map.append({"cpu_ids": cpu_ids, "numa": numa, "is_phys": is_phys})
    return col_map


# ---------------------------------------------------------------------------
# CPU pinning parser (boot log)
# ---------------------------------------------------------------------------

def parse_cpu_pinning(run_dir: Path) -> list[dict]:
    """
    Parse CPU pinning info from hybrid_server_boot.log.
    Correlates:
      - numa_bind_node=N   (from _get_autobind_cpu_ids line)
      - auto thread-binding list [(id, phys), ...]  (separate line)
    Returns list of {engine, numa_bind, cpu_ids, n_threads, cpu_range, is_phys}
    sorted by engine index.
    """
    log_path = run_dir / "hybrid_server_boot.log"
    if not log_path.exists():
        return []

    engine_pat  = re.compile(r"\(CPU_EngineCore_(\d+) pid=\d+\)")
    binding_pat = re.compile(r"auto thread-binding list.*?:\s*(\[.*?\])")
    numa_pat    = re.compile(r"numa_bind_node=(\d+)")

    # First pass: collect numa_bind per engine index
    numa_by_engine: dict[int, int] = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            em = engine_pat.search(line)
            nm = numa_pat.search(line)
            if em and nm:
                numa_by_engine[int(em.group(1))] = int(nm.group(1))

    # Second pass: collect binding list per engine index
    bindings: dict[int, list[int]] = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            em = engine_pat.search(line)
            bm = binding_pat.search(line)
            if em and bm:
                eid = int(em.group(1))
                try:
                    pairs   = eval(bm.group(1))
                    cpu_ids = [p[0] for p in pairs]
                except Exception:
                    cpu_ids = []
                bindings[eid] = cpu_ids

    results = []
    for eid in sorted(set(list(numa_by_engine.keys()) + list(bindings.keys()))):
        cpu_ids  = bindings.get(eid, [])
        numa_bind = numa_by_engine.get(eid, -1)
        is_phys  = None
        if cpu_ids:
            # will be refined with phys_thresh in caller; use heuristic here
            is_phys = all(c < 112 for c in cpu_ids)
        cpu_min = min(cpu_ids) if cpu_ids else 0
        cpu_max = max(cpu_ids) if cpu_ids else 0
        results.append({
            "engine":    f"CPU_EngineCore_{eid}",
            "numa_bind": numa_bind,
            "cpu_ids":   cpu_ids,
            "n_threads": len(cpu_ids),
            "cpu_range": f"{cpu_min}~{cpu_max}",
            "is_phys":   is_phys,
        })
    return results


# ---------------------------------------------------------------------------
# CPU utilization by NUMA
# ---------------------------------------------------------------------------

def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": round(sum(vals) / len(vals), 1),
        "max":  round(max(vals), 1),
        "min":  round(min(vals), 1),
    }


def compute_numa_cpu_util(csv_path: Path, col_map: list[dict]) -> dict:
    """
    Returns:
        {
          numa_node (int): {
            "physical": {"mean", "max", "min"},
            "ht":       {"mean", "max", "min"},
            "combined": {"mean", "max", "min"},
          }
        }
        Also "overall": combined across all NUMA nodes
    """
    if not csv_path.exists():
        return {}

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return {}

    # Identify available core columns
    core_cols = [h for h in rows[0].keys() if re.match(r"core\d+_util_pct$", h)]
    num_cols   = len(core_cols)

    if num_cols != len(col_map):
        # col_map was built with total_cpus//tpc; CSV may differ
        # Rebuild partial map by column index only
        pass

    # Accumulate per-sample averages per category
    # Structure: {numa: {phys: [sample_means], ht: [sample_means]}}
    from collections import defaultdict
    phys_series: dict[int, list[float]] = defaultdict(list)
    ht_series:   dict[int, list[float]] = defaultdict(list)
    all_series:  list[float] = []

    for row in rows:
        # group columns by numa+type → collect values for this sample
        bucket_phys: dict[int, list[float]] = defaultdict(list)
        bucket_ht:   dict[int, list[float]] = defaultdict(list)
        all_vals: list[float] = []

        for col_hdr in core_cols:
            m = re.match(r"core(\d+)_util_pct$", col_hdr)
            if not m:
                continue
            idx = int(m.group(1))
            if idx >= len(col_map):
                continue
            v = row.get(col_hdr, "")
            if v in ("", "N/A"):
                continue
            fv = float(v)
            info = col_map[idx]
            numa = info["numa"]
            if info["is_phys"]:
                bucket_phys[numa].append(fv)
            else:
                bucket_ht[numa].append(fv)
            all_vals.append(fv)

        # One mean per sample per bucket
        for numa, vs in bucket_phys.items():
            phys_series[numa].append(sum(vs) / len(vs))
        for numa, vs in bucket_ht.items():
            ht_series[numa].append(sum(vs) / len(vs))
        if all_vals:
            all_series.append(sum(all_vals) / len(all_vals))

    all_numas = sorted(set(list(phys_series.keys()) + list(ht_series.keys())))
    result: dict = {}
    for numa in all_numas:
        p = phys_series.get(numa, [])
        h = ht_series.get(numa, [])
        combined = p + h  # interleaved — not per-sample combined, but good enough
        # Per-sample combined
        per_sample_combined = []
        for i in range(max(len(p), len(h))):
            vals = []
            if i < len(p): vals.append(p[i])
            if i < len(h): vals.append(h[i])
            per_sample_combined.append(sum(vals) / len(vals))
        result[numa] = {
            "physical": _stats(p),
            "ht":       _stats(h),
            "combined": _stats(per_sample_combined),
            "n_phys_cols": len([c for c in col_map if c["numa"] == numa and c["is_phys"]]),
            "n_ht_cols":   len([c for c in col_map if c["numa"] == numa and not c["is_phys"]]),
        }

    result["overall"] = _stats(all_series)
    result["sample_count"] = len(rows)
    return result


# ---------------------------------------------------------------------------
# GPU utilization
# ---------------------------------------------------------------------------

def compute_gpu_util(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return {}

    headers = list(rows[0].keys())
    gpu_util_cols  = sorted([h for h in headers if re.match(r"gpu\d+_util_pct$", h)])
    gpu_power_cols = sorted([h for h in headers if re.match(r"gpu\d+_power_w$", h)])

    per_gpu = {}
    for col in gpu_util_cols:
        m = re.match(r"gpu(\d+)_util_pct$", col)
        if not m:
            continue
        idx = int(m.group(1))
        vals = [float(r[col]) for r in rows if r.get(col, "") not in ("", "N/A")]
        per_gpu[idx] = _stats(vals)

    per_gpu_power = {}
    for col in gpu_power_cols:
        m = re.match(r"gpu(\d+)_power_w$", col)
        if not m:
            continue
        idx = int(m.group(1))
        vals = [float(r[col]) for r in rows if r.get(col, "") not in ("", "N/A")]
        per_gpu_power[idx] = _stats(vals)

    avg_util_col = "gpu_avg_util_pct"
    avg_vals = [float(r[avg_util_col]) for r in rows if r.get(avg_util_col, "") not in ("", "N/A")]

    return {
        "per_gpu":       per_gpu,
        "per_gpu_power": per_gpu_power,
        "avg":           _stats(avg_vals),
        "sample_count":  len(rows),
        "duration_s":    round(float(rows[-1]["elapsed_s"]) - float(rows[0]["elapsed_s"]), 1) if len(rows) > 1 else 0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(run_dir: Path) -> str:
    si       = load_system_info(run_dir)
    cpu_maps = build_cpu_maps(si)
    cpu_to_numa, total_cpus, tpc, phys_thresh = cpu_maps

    lines = []
    W = 80

    def hr(ch="─"): lines.append(ch * W)
    def title(s):   lines.append(f"  {s}")
    def blank():    lines.append("")

    lines.append("═" * W)
    title("vLLM Hybrid — Hardware Utilization Inspect")
    title(f"Dir  : {run_dir.name}")
    title(f"Time : {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}")
    lines.append("═" * W)
    blank()

    # ── System Info ────────────────────────────────────────────────────────
    hr()
    title("[System]")
    hr()
    cpu_si  = si.get("cpu", {})
    gpu_si  = si.get("gpu", {})
    numa_si = si.get("numa", {})
    title(f"CPU  : {cpu_si.get('model_name', '?')}")
    title(f"       {cpu_si.get('sockets', '?')} sockets × "
          f"{cpu_si.get('cores_per_socket', '?')} cores/socket × "
          f"{cpu_si.get('threads_per_core', '?')} threads = "
          f"{total_cpus} logical CPUs")
    title(f"       phys_threshold={phys_thresh}  "
          f"(CPU < {phys_thresh} → physical primary)")

    gpu_devs = gpu_si.get("devices", [{}])
    gpu_name = gpu_devs[0].get("name", "?") if gpu_devs else "?"
    title(f"GPU  : {gpu_name} × {gpu_si.get('count', '?')}")

    numa_nodes = numa_si.get("nodes", [])
    for nd in numa_nodes:
        node_id  = nd.get("node", "?")
        size_mb  = nd.get("size_mb", "?")
        # List CPUs in this node
        node_cpus = sorted(k for k, v in cpu_to_numa.items() if v == node_id)
        if node_cpus:
            phys = [c for c in node_cpus if c < phys_thresh]
            ht   = [c for c in node_cpus if c >= phys_thresh]
            p_str = f"{min(phys)}~{max(phys)}" if phys else "-"
            h_str = f"{min(ht)}~{max(ht)}"     if ht   else "-"
            title(f"NUMA {node_id}: {size_mb // 1024}GB  "
                  f"phys={p_str}  ht={h_str}")
        else:
            title(f"NUMA {node_id}: {size_mb // 1024 if isinstance(size_mb, int) else '?'}GB")
    blank()

    # ── CPU Pinning ────────────────────────────────────────────────────────
    hr()
    title("[CPU Pinning  — from hybrid_server_boot.log]")
    hr()

    mode = "hybrid" if (run_dir / "hybrid.json").exists() else "gpu_only"
    pinnings = []
    if mode == "hybrid":
        pinnings = parse_cpu_pinning(run_dir)

    if not pinnings:
        if mode == "gpu_only":
            title("  gpu_only mode — no CPU engine pinning")
        else:
            title("  (boot log not found or no pinning info)")
    else:
        for p in pinnings:
            cpu_ids = p["cpu_ids"]
            if cpu_ids:
                phys_count = sum(1 for c in cpu_ids if c < phys_thresh)
                ht_count   = len(cpu_ids) - phys_count
                p["is_phys"] = (phys_count > 0 and ht_count == 0)
                if phys_count > 0 and ht_count == 0:
                    ptype = "physical primary ✓"
                elif ht_count > 0 and phys_count == 0:
                    ptype = "HT sibling ← !"
                else:
                    ptype = f"mixed (phys={phys_count} ht={ht_count})"
            else:
                ptype = "?"

            title(f"  {p['engine']:22s}  NUMA {p['numa_bind']}  "
                  f"CPUs {p['cpu_range']}  ({p['n_threads']} threads, {ptype})")
    blank()

    # ── CPU Utilization ────────────────────────────────────────────────────
    hr()
    title("[CPU Utilization — per NUMA node]")
    hr()

    cpu_csv = run_dir / f"{mode}_monitor_cpu.csv"
    col_map: list[dict] = []
    if cpu_csv.exists() and total_cpus > 0:
        rows_probe = list(csv.DictReader(open(cpu_csv)))
        if rows_probe:
            core_cols_probe = [h for h in rows_probe[0].keys()
                               if re.match(r"core\d+_util_pct$", h)]
            num_cols = len(core_cols_probe)
            col_map  = build_column_map(num_cols, tpc, cpu_to_numa, phys_thresh)

    if col_map:
        numa_util = compute_numa_cpu_util(cpu_csv, col_map)

        all_numas = sorted(k for k in numa_util if isinstance(k, int))
        for numa in all_numas:
            u = numa_util[numa]
            n_phys = u.get("n_phys_cols", 0)
            n_ht   = u.get("n_ht_cols",   0)
            node_cpus = sorted(k for k, v in cpu_to_numa.items() if v == numa)
            phys = [c for c in node_cpus if c < phys_thresh]
            ht   = [c for c in node_cpus if c >= phys_thresh]
            p_str = f"{min(phys)}~{max(phys)}" if phys else "-"
            h_str = f"{min(ht)}~{max(ht)}"     if ht   else "-"

            title(f"  NUMA {numa}  "
                  f"(phys CPUs {p_str} [{n_phys} cols]  "
                  f"ht CPUs {h_str} [{n_ht} cols])")
            sp = u["physical"]
            sh = u["ht"]
            sc = u["combined"]
            title(f"    physical  : mean={sp['mean']:5.1f}%  "
                  f"max={sp['max']:5.1f}%  min={sp['min']:5.1f}%")
            title(f"    HT sibling: mean={sh['mean']:5.1f}%  "
                  f"max={sh['max']:5.1f}%  min={sh['min']:5.1f}%")
            title(f"    combined  : mean={sc['mean']:5.1f}%  "
                  f"max={sc['max']:5.1f}%  min={sc['min']:5.1f}%")
            blank()

        ov = numa_util.get("overall", {})
        sc = numa_util.get("sample_count", 0)
        title(f"  overall: mean={ov.get('mean',0):5.1f}%  "
              f"max={ov.get('max',0):5.1f}%   ({sc} samples)")
    else:
        title("  (no CPU monitor CSV or NUMA info unavailable)")
    blank()

    # ── GPU Utilization ────────────────────────────────────────────────────
    hr()
    title("[GPU Utilization]")
    hr()

    gpu_csv  = run_dir / f"{mode}_monitor_gpu.csv"
    gpu_util = compute_gpu_util(gpu_csv)

    if not gpu_util:
        title("  (no GPU monitor CSV)")
    else:
        per_gpu = gpu_util.get("per_gpu", {})
        pwr     = gpu_util.get("per_gpu_power", {})
        for idx in sorted(per_gpu.keys()):
            u  = per_gpu[idx]
            pw = pwr.get(idx, {})
            title(f"  GPU{idx}: util mean={u['mean']:5.1f}%  "
                  f"max={u['max']:5.1f}%  "
                  f"| power mean={pw.get('mean',0):6.0f}W  "
                  f"max={pw.get('max',0):6.0f}W")
        avg = gpu_util.get("avg", {})
        sc  = gpu_util.get("sample_count", 0)
        dur = gpu_util.get("duration_s", 0)
        blank()
        title(f"  avg: mean={avg.get('mean',0):5.1f}%  "
              f"max={avg.get('max',0):5.1f}%  "
              f"({sc} samples, {dur:.0f}s)")
    blank()

    lines.append("═" * W)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect hardware utilization of a single benchmark run",
        usage="%(prog)s <result_dir>",
    )
    parser.add_argument("result_dir", help="Benchmark result directory")
    args = parser.parse_args()

    run_dir = Path(args.result_dir)
    if not run_dir.is_dir():
        print(f"[ERROR] Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    report = build_report(run_dir)
    print(report)

    out_path = run_dir / "inspect.txt"
    out_path.write_text(report)
    print(f"\n[inspect] Saved → {out_path}")


if __name__ == "__main__":
    main()
