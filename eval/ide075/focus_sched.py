#!/usr/bin/env python3
"""IDE_075 §12.7/§11.5 — FOCUS 세션: perf sched_switch/wakeup 로 tail 작업(서비스 >2.5 ms 또는 NUMA 시차 >1 ms) 의 off-CPU 시간을 귀속.
perf 시각(CLOCK_MONOTONIC 초) → REALTIME: clock_anchors.jsonl 의 (realtime_ns, monotonic_ns) 단일 anchor (세션 30~40 s, drift 무시; 한계 기록).
대상 TID: 스레드 이름 기반 (kt-task-worker, KT NUMA worker = thread_role_map 의 affinity 가 kt cpu 1~2개인 스레드).
각 tail 작업의 [t_numa_start, t_numa_end] (numa0/1) 안에서 NUMA worker TID 들이 off-CPU(sched_switch prev_state != R, 다음 switch-in 까지) 였던 시간의 합·최대, 그리고 worker 가 아닌 스레드가 kt cpu 를 점유한 시간을 집계.
산출: <session>/v2/tail_offcpu.csv, tail_offcpu_summary.json"""
import json, os, sys, csv, subprocess, collections, re, bisect
HOME = os.path.expanduser("~"); KT = f"{HOME}/.cache/huggingface/kt/ide075"


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]; out = f"{sd}/v2"; os.makedirs(out, exist_ok=True)
    anc = [json.loads(l) for l in open(f"{sd}/clock_anchors.jsonl")][0]; off_ns = anc["realtime_ns"] - anc["monotonic_ns"]
    data = f"{sd}/perf_sched.data"
    if not os.path.exists(f"{sd}/perf_sched.txt") or os.path.getsize(f"{sd}/perf_sched.txt") == 0:
        subprocess.run(f"sudo perf script -f -i {data} > {sd}/perf_sched.txt 2>{sd}/perf_script.err", shell=True); subprocess.run(f"sudo chown $(id -u) {sd}/perf_sched.txt {sd}/perf_script.err", shell=True)
    # 파싱: "comm tid [cpu] time: event: prev_comm=... prev_pid=... prev_state=... ==> next_comm=... next_pid=..."
    sw = []   # (t_ns, cpu, prev_pid, prev_state, next_pid)
    rx = re.compile(r"^.*?\s(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+sched:sched_switch:\s+(.*)$")
    for l in open(f"{sd}/perf_sched.txt", errors="replace"):
        mm = rx.match(l)
        if not mm: continue
        t = int(float(mm.group(3)) * 1e9) + off_ns; tr = mm.group(4)
        # 두 형식 지원: "prev_comm=.. prev_pid=N .. prev_state=S ==> next_pid=M" / "comm:N [prio] S ==> comm:M [prio]" (perf 6.x 기본)
        pp = re.search(r"prev_pid=(\d+)", tr); ps = re.search(r"prev_state=(\S+)", tr); npid = re.search(r"next_pid=(\d+)", tr)
        if not (pp and npid):
            m2 = re.match(r"^(.*):(\d+) \[\d+\] (\S+) ==> (.*):(\d+) \[\d+\]", tr)
            if m2: sw.append((t, int(mm.group(2)), int(m2.group(2)), m2.group(3), int(m2.group(5)))); continue
        if pp and npid: sw.append((t, int(mm.group(2)), int(pp.group(1)), ps.group(1) if ps else "?", int(npid.group(1))))
    sw.sort()
    # TID 별 off-CPU 구간: switch-out(prev_pid=T) → 다음 switch-in(next_pid=T)
    outs = collections.defaultdict(list); last_out = {}; n_switch_in = collections.Counter(); switch_outs = collections.defaultdict(list)
    for t, cpu, prev, state, nxt in sw:
        last_out[prev] = (t, state); switch_outs[prev].append((t, state, cpu))
        if nxt in last_out:
            t0, st = last_out.pop(nxt); outs[nxt].append((t0, t, st)); n_switch_in[nxt] += 1
    # 역할 TID
    trm = f"{os.path.dirname(sd.rstrip('/'))}/thread_role_map.csv"; roles = {}
    kt = set(json.load(open(f"{os.path.dirname(sd.rstrip('/'))}/../../../state/RUN_STATE.json"))["boots"][boot]["kt_cpus"])
    for r in csv.DictReader(open(trm)):
        a = r.get("affinity_list", ""); cs = set()
        for part in a.split(","):
            try:
                if "-" in part: x, y = part.split("-"); cs |= set(range(int(x), int(y) + 1))
                elif part.strip(): cs.add(int(part))
            except ValueError: pass
        roles[int(r["tid"])] = r["thread_name"] if r["thread_name"].startswith("kt-") else ("KT_NUMA_worker" if ((cs and cs <= kt and len(cs) <= 2) or re.match(r"numa_\d+_t_\d+", r["thread_name"] or "")) else "other")
    workers = [t for t, ro in roles.items() if ro == "KT_NUMA_worker"]
    # tail 작업 (kt_evt 세션 창)
    t0, t1 = m["t_start"]["epoch"] * 1e9, m["drain"]["t_drain_end"]["epoch"] * 1e9
    ev = [r for r in csv.DictReader(l for l in open(f"{KT}/{boot}/kt_evt.csv") if not l.startswith("#")) if t0 <= int(r["t_go"]) <= t1 and int(r["t_fwd_exit"] or 0)]
    dec = [r for r in ev if int(r.get("qlen") or 0) <= 64]; pre = [r for r in ev if int(r.get("qlen") or 0) > 64]
    tails = [r for r in dec if (int(r["t_fwd_exit"]) - int(r["t_fwd_entry"])) > 2_500_000 or abs(int(r["t_numa0_end"] or 0) - int(r["t_numa1_end"] or 0)) > 1_000_000]   # 디코드(qlen ≤ 64) 한정; prefill 은 별도
    tails_pre = [r for r in pre if (int(r["t_fwd_exit"]) - int(r["t_fwd_entry"])) > 12_000_000 or abs(int(r["t_numa0_end"] or 0) - int(r["t_numa1_end"] or 0)) > 1_000_000]
    rows = []; wset = set(workers); w_in = sum(n_switch_in[w] for w in workers); w_out = sum(len(switch_outs[w]) for w in workers)
    intervals_recoverable = w_in > 0   # -p 기록은 switch-in 이 없어 구간 길이 산출 불가
    sw_lo, sw_hi = (sw[0][0], sw[-1][0]) if sw else (0, 0)
    for r in tails + tails_pre:
        for s in (0, 1):
            a, b = int(r[f"t_numa{s}_start"] or 0), int(r[f"t_numa{s}_end"] or 0)
            if not a or not b: continue
            off_sum = 0; off_max = 0; n_off = 0; states = collections.Counter(); n_out = 0; out_states = collections.Counter()
            for w in workers:
                for (o0, o1, st) in outs.get(w, []):
                    x0, x1 = max(o0, a), min(o1, b)
                    if x1 > x0: off_sum += x1 - x0; off_max = max(off_max, x1 - x0); n_off += 1; states[st] += 1
                for (t_, st, cpu) in switch_outs.get(w, []):
                    if a <= t_ <= b: n_out += 1; out_states[st] += 1
            rows.append({"slot": r["slot"], "epoch": r["epoch"], "kind": "decode" if int(r.get("qlen") or 0) <= 64 else "prefill", "qlen": r.get("qlen"), "numa": s, "span_us": (b - a) / 1e3, "service_us": (int(r["t_fwd_exit"]) - int(r["t_fwd_entry"])) / 1e3, "skew_us": abs(int(r["t_numa0_end"]) - int(r["t_numa1_end"])) / 1e3, "worker_offcpu_sum_us": off_sum / 1e3 if intervals_recoverable else None, "worker_offcpu_max_us": off_max / 1e3 if intervals_recoverable else None, "n_offcpu_intervals": n_off if intervals_recoverable else None, "prev_states": json.dumps(dict(states)), "n_worker_switch_outs_in_window": n_out, "switch_out_states": json.dumps(dict(out_states)), "perf_covered": int(sw_lo <= a and b <= sw_hi), "n_workers": len(workers)})
    with open(f"{out}/tail_offcpu.csv", "w", newline="") as f:
        if rows: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    err = open(f"{sd}/perf_script.err", errors="replace").read() if os.path.exists(f"{sd}/perf_script.err") else ""
    lost = re.findall(r"lost ([\d.]+)%", err)
    cov_rows = [r for r in rows if r["perf_covered"] and r["kind"] == "decode"]; pre_rows = [r for r in rows if r["kind"] == "prefill"]
    preemptors = collections.Counter()
    for t, cpu, prev, state, nxt in sw:
        if prev in wset: preemptors[nxt] += 1
    def _q(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None
    summ = {"decode_records": len(dec), "prefill_records": len(pre), "tails": len(tails), "tails_prefill(service>12ms|skew>1ms)": len(tails_pre), "rows": len(rows),
            "decode_tail_offcpu_sum_us_p50_p90_max": [_q([r["worker_offcpu_sum_us"] or 0 for r in cov_rows], .5), _q([r["worker_offcpu_sum_us"] or 0 for r in cov_rows], .9), max((r["worker_offcpu_sum_us"] or 0 for r in cov_rows), default=None)],
            "decode_tail_offcpu_max_us_p50_p90_max": [_q([r["worker_offcpu_max_us"] or 0 for r in cov_rows], .5), _q([r["worker_offcpu_max_us"] or 0 for r in cov_rows], .9), max((r["worker_offcpu_max_us"] or 0 for r in cov_rows), default=None)],
            "decode_tail_rows_with_offcpu": sum(1 for r in cov_rows if (r["worker_offcpu_sum_us"] or 0) > 0), "decode_tail_rows_offcpu_gt_300us": sum(1 for r in cov_rows if (r["worker_offcpu_max_us"] or 0) > 300),
            "prefill_tail_offcpu_max_us_max": max((r["worker_offcpu_max_us"] or 0 for r in pre_rows), default=None), "rows_inside_perf_coverage": len(cov_rows), "workers_identified": len(workers), "sched_switch_events": len(sw), "perf_switch_window_realtime_ns": [sw_lo, sw_hi], "perf_lost_samples_pct": lost, "perf_warnings": err.strip()[:300],
            "worker_switch_outs_total": w_out, "worker_switch_ins_total": w_in, "worker_switch_out_states": dict(collections.Counter(st for w in workers for (_, st, _) in switch_outs.get(w, []))), "worker_switch_outs_per_worker_per_s": (w_out / max(len(workers), 1)) / max((sw_hi - sw_lo) / 1e9, 1e-9) if sw else None,
            "offcpu_intervals_recoverable": intervals_recoverable, "offcpu_note": "perf -p 기록: 대상 스레드의 switch-out 만 기록되고 switch-in 이 없어 off-CPU 길이 NOT_RECOVERABLE (횟수·상태·선점자만)" if not intervals_recoverable else "system-wide 기록: switch-out→switch-in 구간",
            "tails_with_worker_switch_out": sum(1 for r in cov_rows if r["n_worker_switch_outs_in_window"] > 0), "switch_outs_in_tail_windows": sum(r["n_worker_switch_outs_in_window"] for r in cov_rows),
            "offcpu_max_us_over_tails": max((r["worker_offcpu_max_us"] for r in rows if r["worker_offcpu_max_us"] is not None), default=None), "tails_with_offcpu_gt_1ms": sum(1 for r in rows if (r["worker_offcpu_max_us"] or 0) > 1000), "decode_tails_with_offcpu_gt_1ms": sum(1 for r in cov_rows if (r["worker_offcpu_max_us"] or 0) > 1000), "clock": f"perf monotonic → realtime 단일 anchor offset {off_ns} ns (drift 미보정)", "note": "off-CPU = sched_switch(prev=worker) ~ 다음 switch-in(next=worker). prev_state R = 선점(runnable), S/D = 대기"}
    json.dump(summ, open(f"{out}/tail_offcpu_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__": main(sys.argv[1])
