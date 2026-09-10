#!/usr/bin/env python3
"""dram_trace.txt (10초 창 샘플) 를 E12 의 구간 경계에 맞춰 구간별 물리 DRAM 평균과
출력 토큰당 바이트를 계산한다. 구간 경계는 각 bench 로그의 시작 시각과 지속시간으로 잡는다.

usage: align_dram.py <E12 결과 디렉토리>
"""
import re, sys, os, glob, datetime

B = sys.argv[1]
NUM = r"([0-9]+\.?[0-9]*)"

def parse_trace(path):
    out = []
    for line in open(path):
        m = re.match(r"(\d\d:\d\d:\d\d) S0 r=\s*([\d.]+) w=\s*([\d.]+) GB/s \| S1 r=\s*([\d.]+) w=\s*([\d.]+)", line)
        if m:
            t = datetime.datetime.strptime(m.group(1), "%H:%M:%S").time()
            out.append((t, float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    return out

def bench_window(path):
    """bench 로그에서 시작 시각 (INFO 첫 줄) 과 지속시간을 뽑아 (시작, 끝) 을 만든다."""
    txt = open(path, errors="ignore").read()
    m = re.search(r"INFO (\d\d)-(\d\d) (\d\d:\d\d:\d\d)", txt)
    d = re.search(r"Benchmark duration \(s\):\s*"+NUM, txt)
    tok = re.search(r"Output token throughput \(tok/s\):\s*"+NUM, txt)
    if not (m and d):
        return None
    t0 = datetime.datetime.strptime(m.group(3), "%H:%M:%S")
    # 컨테이너 시계가 호스트와 다를 수 있으므로 오프셋을 인자로 받는다 (기본 0)
    return t0, float(d.group(1)), float(tok.group(1)) if tok else None

trace = parse_trace(os.path.join(B, "dram_trace.txt"))
print(f"DRAM 샘플 {len(trace)}개 ({trace[0][0]}~{trace[-1][0]})\n")
print("| 구간 | 창 (호스트시계) | 샘플 | S0 읽기 | S1 읽기 | 읽기 합 | 쓰기 합 | 출력 tok/s | 토큰당 읽기 |")
print("|---|---|---|---|---|---|---|---|---|")
OFF = float(os.environ.get("CLOCK_OFFSET_S", "0"))
for f in sorted(glob.glob(os.path.join(B, "soak_*.log"))) + sorted(glob.glob(os.path.join(B, "rep_*.log"))):
    w = bench_window(f)
    if not w:
        continue
    t0, dur, tok = w
    t0 = t0 + datetime.timedelta(seconds=OFF)
    t1 = t0 + datetime.timedelta(seconds=dur)
    # 램프 배제: 짧은 run 은 비례해서 (최대 60초, 최소 지속시간의 25%)
    ramp = min(60.0, 0.25 * dur)
    lo = (t0 + datetime.timedelta(seconds=ramp)).time(); hi = t1.time()
    sel = [x for x in trace if lo <= x[0] <= hi]
    if not sel:
        continue
    n = len(sel)
    s0r = sum(x[1] for x in sel)/n; s1r = sum(x[3] for x in sel)/n
    s0w = sum(x[2] for x in sel)/n; s1w = sum(x[4] for x in sel)/n
    rd = s0r + s1r; wr = s0w + s1w
    per = (rd*1e9/tok/1e6) if tok else None
    print(f"| {os.path.basename(f)[:-4]} | {lo}~{hi} | {n} | {s0r:.1f} | {s1r:.1f} | {rd:.1f} | {wr:.1f} | "
          f"{tok if tok else '-'} | {f'{per:.0f} MB' if per else '-'} |")
