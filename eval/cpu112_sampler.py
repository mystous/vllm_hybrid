#!/usr/bin/env python3
"""Per-core + per-thread CPU sampler + NUMA placement tracker.

Sampling 도구 (mpstat/pidstat 부재 환경):
- /proc/stat per-core jiffies → 1Hz util time-series (224 logical core)
- /proc/<pid>/task/<tid>/stat → per-thread CPU time + state
- /proc/<pid>/task/<tid>/status → Cpus_allowed_list (affinity)
- /proc/<pid>/numa_maps → NUMA placement
- /proc/<pid>/sched → CPU migration count

Usage: python3 cpu112_sampler.py <OUTDIR> <DURATION_SEC> <PIDS_COMMA_SEP>
"""
import os
import sys
import time
import json
from pathlib import Path


def read_proc_stat():
    """/proc/stat 의 cpu<N> line 을 dict 로 반환 — jiffies (user, nice, system, idle, iowait, irq, softirq, steal)."""
    cores = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            if name == "cpu":
                continue
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            cid = int(name[3:])
            jiffs = list(map(int, parts[1:8]))  # user nice sys idle iowait irq softirq
            cores[cid] = jiffs
    return cores


def diff_jiffies(prev, curr):
    """차분 → utilization breakdown."""
    out = {}
    for cid, j in curr.items():
        if cid not in prev:
            continue
        d = [c - p for c, p in zip(j, prev[cid])]
        total = sum(d)
        if total == 0:
            continue
        u, n, s, i, w, ir, si = d
        active = u + n + s + ir + si
        out[cid] = {
            "util_pct": 100.0 * active / total,
            "user_pct": 100.0 * (u + n) / total,
            "sys_pct": 100.0 * s / total,
            "iowait_pct": 100.0 * w / total,
            "idle_pct": 100.0 * i / total,
            "total_jiffies": total,
        }
    return out


def list_threads(pid):
    """pid 의 thread list."""
    p = Path(f"/proc/{pid}/task")
    if not p.exists():
        return []
    return sorted(int(t.name) for t in p.iterdir() if t.name.isdigit())


def read_thread_stat(pid, tid):
    """/proc/<pid>/task/<tid>/stat — (comm, state, utime, stime, last_cpu)."""
    try:
        with open(f"/proc/{pid}/task/{tid}/stat") as f:
            data = f.read()
        # comm 안 공백 처리: (...) 닫는 ) 기준
        rpar = data.rfind(")")
        comm = data[data.find("(") + 1 : rpar]
        rest = data[rpar + 2 :].split()
        state = rest[0]
        utime = int(rest[11])
        stime = int(rest[12])
        last_cpu = int(rest[36]) if len(rest) > 36 else -1
        return comm, state, utime, stime, last_cpu
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None


def read_thread_affinity(pid, tid):
    """/proc/<pid>/task/<tid>/status 의 Cpus_allowed_list."""
    try:
        with open(f"/proc/{pid}/task/{tid}/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def read_wchan(pid, tid):
    """/proc/<pid>/task/<tid>/wchan — what the thread is waiting on."""
    try:
        with open(f"/proc/{pid}/task/{tid}/wchan") as f:
            return f.read().strip() or "0"
    except (FileNotFoundError, ProcessLookupError):
        return None


def read_numa_maps_summary(pid):
    """/proc/<pid>/numa_maps — node 별 page 합산."""
    try:
        node_pages = {}
        with open(f"/proc/{pid}/numa_maps") as f:
            for line in f:
                # 형식: addr policy mapped_file? N<id>=<pages> N<id2>=<pages2> ...
                for tok in line.split():
                    if tok.startswith("N") and "=" in tok:
                        k, v = tok[1:].split("=")
                        if k.isdigit():
                            node_pages[int(k)] = node_pages.get(int(k), 0) + int(v)
        return node_pages
    except (FileNotFoundError, ProcessLookupError):
        return None


def read_numastat(pid):
    """/proc/<pid>/status — VmRSS + numa accounting."""
    try:
        out = {}
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmSize:", "VmPeak:", "voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:")):
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
        return out
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    if len(sys.argv) < 4:
        print("usage: cpu112_sampler.py <OUTDIR> <DURATION_SEC> <PIDS_COMMA_SEP>", file=sys.stderr)
        sys.exit(1)
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    duration = int(sys.argv[2])
    pids_arg = sys.argv[3]
    target_pids = [int(p) for p in pids_arg.split(",") if p.strip()]

    # output files
    core_log = (outdir / "per_core_util.jsonl").open("w")
    thread_log = (outdir / "per_thread_stat.jsonl").open("w")
    numa_log = (outdir / "numa_placement.jsonl").open("w")
    affinity_log = (outdir / "thread_affinity.jsonl").open("w")
    wchan_log = (outdir / "thread_wchan.jsonl").open("w")

    # 사전 jiffies
    prev = read_proc_stat()
    prev_thread = {}  # (pid, tid) -> (utime, stime)
    t_start = time.time()

    # affinity / numa 는 30 s 간격 (cheap)
    last_affinity = 0
    last_numa = 0

    sample_idx = 0
    while time.time() - t_start < duration:
        t0 = time.time()
        time.sleep(1.0)
        t1 = time.time()
        sample_idx += 1

        # per-core util
        curr = read_proc_stat()
        util = diff_jiffies(prev, curr)
        prev = curr
        rec = {"t": t1, "idx": sample_idx, "cores": util}
        core_log.write(json.dumps(rec) + "\n")
        core_log.flush()

        # per-thread (sampled per-second)
        thread_rec = {"t": t1, "idx": sample_idx, "threads": {}}
        for pid in target_pids:
            for tid in list_threads(pid):
                s = read_thread_stat(pid, tid)
                if s is None:
                    continue
                comm, state, utime, stime, last_cpu = s
                prev_t = prev_thread.get((pid, tid))
                if prev_t is not None:
                    dut = utime - prev_t[0]
                    dst = stime - prev_t[1]
                else:
                    dut = dst = 0
                prev_thread[(pid, tid)] = (utime, stime)
                thread_rec["threads"][f"{pid}/{tid}"] = {
                    "comm": comm,
                    "state": state,
                    "d_utime": dut,
                    "d_stime": dst,
                    "last_cpu": last_cpu,
                }
        thread_log.write(json.dumps(thread_rec) + "\n")
        thread_log.flush()

        # wchan (어디서 기다리는지) — 매 sample
        wchan_rec = {"t": t1, "idx": sample_idx, "wchan": {}}
        for pid in target_pids:
            for tid in list_threads(pid):
                w = read_wchan(pid, tid)
                if w and w != "0":
                    wchan_rec["wchan"][f"{pid}/{tid}"] = w
        wchan_log.write(json.dumps(wchan_rec) + "\n")
        wchan_log.flush()

        # affinity (30 s 마다)
        if t1 - last_affinity > 30:
            aff_rec = {"t": t1, "idx": sample_idx, "affinity": {}}
            for pid in target_pids:
                for tid in list_threads(pid):
                    a = read_thread_affinity(pid, tid)
                    if a:
                        aff_rec["affinity"][f"{pid}/{tid}"] = a
            affinity_log.write(json.dumps(aff_rec) + "\n")
            affinity_log.flush()
            last_affinity = t1

        # numa placement (30 s 마다)
        if t1 - last_numa > 30:
            numa_rec = {"t": t1, "idx": sample_idx, "numa": {}, "status": {}}
            for pid in target_pids:
                np_ = read_numa_maps_summary(pid)
                if np_:
                    numa_rec["numa"][str(pid)] = np_
                st = read_numastat(pid)
                if st:
                    numa_rec["status"][str(pid)] = st
            numa_log.write(json.dumps(numa_rec) + "\n")
            numa_log.flush()
            last_numa = t1

    for f in [core_log, thread_log, numa_log, affinity_log, wchan_log]:
        f.close()
    print(f"[sampler] done. {sample_idx} samples. outdir={outdir}")


if __name__ == "__main__":
    main()
