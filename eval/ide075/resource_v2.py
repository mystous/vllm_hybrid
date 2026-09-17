#!/usr/bin/env python3
"""IDE_075 A07 — RESOURCE 세션의 부하 창(forward_envelope = kt_evt first_go ~ last_deferred_end) 안 CPU 카운터·PCM 집계.
perf: `perf stat -I 1000 -x ,` interval 행 (상대 초, 시작 = collector actual_start epoch) → 창 완전 포함 interval 의 합/평균 (경계 interval 은 boundary 계수, 비례 환산 없음).
PCM: raw csv, tsparse.local_timestamp_ns(Asia/Seoul), 표본 구간 [ts−1 s, ts) ASSUMED_END_TIMESTAMP, classify_samples.
산출: <session>/v2/resource_summary.json, <session>/v2/cpu_counter_windows.csv, <session>/v2/pcm_samples_v2.csv"""
import json, os, sys, csv, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsparse import local_timestamp_ns, classify_samples, epoch_float_to_ns


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); out = f"{sd}/v2"; os.makedirs(out, exist_ok=True)
    fg, ld = m.get("first_go_ns"), m.get("last_deferred_end_ns"); W = (fg, ld) if fg and ld else None
    res = {"session": m["session"], "window": {"name": "forward_envelope(kt_evt)", "start_ns": fg, "end_ns": ld, "len_s": (ld - fg) / 1e9 if W else None, "status": "OK" if W else "NOT_COLLECTED"}, "client_process_window_s": m["wall_seconds"], "benchmark_duration_s": (m.get("summary") or {}).get("duration")}
    # ---- perf intervals ----
    pc = next((c for c in m["observer"]["collectors"] if c["collector_id"] == "perf_stat_interval"), None); pf = f"{sd}/cpu_counter_intervals.csv"
    if pc and os.path.exists(pf):
        t0 = epoch_float_to_ns(pc["actual_start"]["epoch"]); iv = collections.defaultdict(dict); bad = 0
        for l in open(pf):
            if not l.strip() or l.startswith("#"): continue
            p = l.strip().split(",")
            try: t = float(p[0]); v = p[1]; ev = p[3] if len(p) > 3 else ""
            except Exception: bad += 1; continue
            if "<not" in v: iv[t][ev] = None; continue
            try: iv[t][ev] = float(v)
            except ValueError: bad += 1
        rows = []
        for t in sorted(iv):
            a = t0 + int(round((t - 1.0) * 1e9)); b = t0 + int(round(t * 1e9)); cls = "unknown" if not W else ("full" if a >= W[0] and b <= W[1] else ("outside" if b <= W[0] or a >= W[1] else "boundary"))
            rows.append({"interval_end_rel_s": t, "start_ns": a, "end_ns": b, "class": cls, **{k: v for k, v in iv[t].items()}})
        with open(f"{out}/cpu_counter_windows.csv", "w", newline="") as f:
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys: keys.append(k)
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        full = [r for r in rows if r["class"] == "full"]
        def s(k): v = [r.get(k) for r in full if r.get(k) is not None]; return sum(v) if v else None
        tc = s("task-clock"); cyc = s("cycles"); ins = s("instructions")
        res["perf"] = {"intervals_total": len(rows), "full": len(full), "boundary": sum(r["class"] == "boundary" for r in rows), "outside": sum(r["class"] == "outside" for r in rows), "bad_rows": bad, "full_window_s": len(full) * 1.0,
                       "task_clock_msec_sum_full": tc, "cpu_equivalents_full": (tc / 1000.0 / len(full)) if (tc and full) else None, "cycles_sum": cyc, "instructions_sum": ins, "ipc_full": (ins / cyc) if (cyc and ins) else None, "context_switches_sum": s("context-switches"), "cpu_migrations_sum": s("cpu-migrations"), "cache_misses_sum": s("cache-misses"),
                       "scope": pc.get("target_scope"), "target_pids": pc.get("target"), "interval_contract": "perf -I 1000: 각 행 = 직전 1 s 구간 (상대 초, 시작 = collector actual_start)", "note": "task-clock 은 논리 CPU 시간 합 (busy-poll 포함); 물리코어 수 아님. 역할별 분리는 thread_role_map 대조 필요 (프로세스 집계)"}
    # ---- PCM ----
    pcm = f"{sd}/pcm_memory.raw.csv"
    if os.path.exists(pcm):
        rows = list(csv.reader(open(pcm, errors="replace"))); hdr = [f"{a.strip()}|{b.strip()}" for a, b in zip(rows[0], rows[1])]
        ix = {k: [i for i, h in enumerate(hdr) if h == k] for k in ("System|Read", "System|Write", "System|Memory", "SKT0|Memory", "SKT1|Memory")}
        di = hdr.index("|Date"); ti = hdr.index("|Time"); samples = {k: [] for k in ix}; bad = 0; PS = []
        for r in rows[2:]:
            if len(r) < len(hdr) - 1: bad += 1; continue
            try: ts = local_timestamp_ns(r[di], r[ti], "Asia/Seoul")
            except ValueError: bad += 1; continue
            for k, idx in ix.items():
                if idx:
                    try: v = float(r[idx[-1]])
                    except ValueError: continue
                    samples[k].append(((ts - 1_000_000_000, ts), v)); PS.append({"metric": k, "sample_start_ns": ts - 1_000_000_000, "sample_end_ns": ts, "value_MBps": v})
        with open(f"{out}/pcm_samples_v2.csv", "w", newline="") as f:
            if PS: w = csv.DictWriter(f, fieldnames=list(PS[0].keys())); w.writeheader(); w.writerows(PS)
        res["pcm"] = {"invalid_rows": bad, "interval_contract": "ASSUMED_END_TIMESTAMP (1 s)", "unit": "MB/s", "by_metric": {}}
        if W:
            for k in ix:
                if samples[k]: res["pcm"]["by_metric"][k] = classify_samples(samples[k], W)
        else: res["pcm"]["status"] = "NOT_COLLECTED(window)"
    json.dump(res, open(f"{out}/resource_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"window_s": res["window"]["len_s"], "perf": {k: res.get("perf", {}).get(k) for k in ("full", "boundary", "cpu_equivalents_full", "ipc_full")}, "pcm_read_full": (res.get("pcm", {}).get("by_metric", {}).get("System|Read") or {}).get("mean_full")}))


if __name__ == "__main__": main(sys.argv[1])
