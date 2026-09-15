#!/usr/bin/env python3
"""IDE_068 셀 디렉터리 → 마크다운 표. 원본(bench.log, cpu_util.txt, hbm_after_boot.csv,
free_after_boot.txt, verdict.txt, smoke_texts.txt)에서만 읽는다.

사용: summarize.py <results_dir> [<results_dir> ...]
"""
from __future__ import annotations

import os
import re
import statistics as st
import sys

FIELDS = [("Successful requests", "완료"), ("Failed requests", "실패"),
          ("Benchmark duration (s)", "소요 s"),
          ("Output token throughput (tok/s)", "출력 tok/s"),
          ("Total token throughput (tok/s)", "전체 tok/s"),
          ("Median TTFT (ms)", "TTFT p50"), ("P95 TTFT (ms)", "TTFT p95"),
          ("Median TPOT (ms)", "TPOT p50"), ("P95 TPOT (ms)", "TPOT p95")]
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def bench(p):
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, errors="replace"):
        line = ANSI.sub("", line).split("\r")[-1].strip()
        for k, _ in FIELDS:
            if line.startswith(k):
                out[k] = line[len(k):].strip(": ").strip()
    return out


def cpu(p):
    if not os.path.exists(p):
        return None
    v = [float(l.split("busy=")[1]) for l in open(p) if "busy=" in l]
    return (sum(v) / len(v), max(v), len(v)) if v else None


def hbm(p):
    if not os.path.exists(p):
        return None
    used = []
    for line in open(p):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            try:
                u = float(parts[1].split()[0])
                if u > 100:
                    used.append(u / 1024)
            except ValueError:
                pass
    return used


def dram(p):
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith("Mem:"):
            return int(line.split()[2])
    return None


def smoke_ok(p):
    if not os.path.exists(p):
        return "—"
    t = open(p, errors="replace").read()
    q = {}
    for i, blk in enumerate(re.split(r"--- Q\d+ ---", t)[1:], 1):
        q[i] = blk
    checks = [("Paris" in q.get(1, "")),
              ("fibonacci(n-1)" in q.get(2, "") or "fibonacci(n - 1)" in q.get(2, "")
               or "a, b = b, a + b" in q.get(2, "")),
              ("5050" in q.get(3, "") or "n(n+1)/2" in q.get(3, "").replace(" ", "")
               or "arithmetic" in q.get(3, "").lower() or "100" in q.get(3, "")),
              ("[::-1]" in q.get(4, "") or "reversed(" in q.get(4, ""))]
    return f"{sum(checks)}/4"


def main(dirs):
    rows = []
    for d in dirs:
        for cell in sorted(os.listdir(d)):
            cp = os.path.join(d, cell)
            if not os.path.isdir(cp) or not os.path.exists(os.path.join(cp, "verdict.txt")):
                continue
            v = open(os.path.join(cp, "verdict.txt")).read().split()
            h = hbm(os.path.join(cp, "hbm_after_boot.csv"))
            dr = dram(os.path.join(cp, "free_after_boot.txt"))
            smoke = smoke_ok(os.path.join(cp, "smoke_texts.txt"))
            # 벤치가 셀 루트에 있거나 (단일 C) c<N>/ 하위에 있다 (C sweep)
            subs = sorted([x for x in os.listdir(cp) if re.fullmatch(r"c\d+", x)],
                          key=lambda x: int(x[1:]))
            targets = [(cell, cp)] if not subs else [(f"{cell} @{x}", os.path.join(cp, x)) for x in subs]
            for label, bp in targets:
                b = bench(os.path.join(bp, "bench.log"))
                c = cpu(os.path.join(bp, "cpu_util.txt"))
                rows.append({
                    "run": os.path.basename(d.rstrip("/")), "cell": label,
                    "verdict": v[0], "boot_s": v[1] if len(v) > 1 else "",
                    "smoke": smoke,
                    **{lab: b.get(k, "—") for k, lab in FIELDS},
                    "cpu": f"{c[0]:.1f} / {c[1]:.1f}" if c else "—",
                    "hbm": (f"{min(h):.1f}~{max(h):.1f} ×{len(h)}" if h else "—"),
                    "dram": f"{dr}" if dr is not None else "—",
                })
    cols = ["cell", "verdict", "boot_s", "smoke", "완료", "출력 tok/s", "전체 tok/s",
            "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95", "소요 s", "cpu", "hbm", "dram"]
    hdr = ["셀", "부팅", "부팅 s", "greedy", "완료", "출력 tok/s", "전체 tok/s",
           "TTFT p50 ms", "TTFT p95 ms", "TPOT p50 ms", "TPOT p95 ms", "벤치 s",
           "CPU busy 평균/최대 %", "HBM GiB/장", "DRAM used GB"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
