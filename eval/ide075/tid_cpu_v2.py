#!/usr/bin/env python3
"""IDE_075 A07 — TID 별 CPU 시간 (cpu_time_by_tid.csv, 1 s 표본) 을 부하 창(forward_envelope=kt_evt first_go~last_deferred_end) 안에서 역할별로 집계.
역할: comm 이름(kt-cf-poll / kt-evt-flush / kt-task-worker: v3 스레드 이름) + affinity(thread_role_map.csv) + CPU 사용 패턴으로 분류. 확정 근거가 없는 스레드는 unknown.
cpu_equivalents_role = Σ(ticks delta within W)/CLK_TCK / |W|. 산출: <session>/v2/cpu_time_by_role.json"""
import json, os, sys, csv, collections
HOME = os.path.expanduser("~")


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); fg, ld = m.get("first_go_ns"), m.get("last_deferred_end_ns")
    p = f"{sd}/cpu_time_by_tid.csv"
    if not (fg and ld and os.path.exists(p)): print("NOT_COLLECTED"); return
    hz = 100; rows = []
    for l in open(p):
        if l.startswith("#clk_tck="): hz = int(l.strip().split("=")[1]); continue
        if l.startswith("epoch_ns"): continue
        f = l.strip().split(",");
        if len(f) < 7: continue
        rows.append((int(f[0]), f[1], f[2], f[3], int(f[4]), int(f[5]) + int(f[6])))
    by = collections.defaultdict(list)
    for t, pid, tid, comm, psr, ticks in rows: by[(pid, tid)].append((t, comm, psr, ticks))
    aff = {}
    trm = f"{os.path.dirname(sd.rstrip('/'))}/thread_role_map.csv"
    if os.path.exists(trm):
        for r in csv.DictReader(open(trm)): aff[r["tid"]] = r.get("affinity_list", "").strip()
    kt = json.load(open(f"{os.path.dirname(sd.rstrip('/'))}/../../../state/RUN_STATE.json")).get("boots", {}).get(m["boot_id"], {}).get("kt_cpus")
    ktset = set(kt or [])
    def role(comm, tid, cpus):
        if comm == "kt-cf-poll": return "poller"
        if comm == "kt-evt-flush": return "observer_flusher"
        if comm == "kt-task-worker": return "task_worker"
        a = aff.get(tid, "")
        if a and ktset:
            try:
                cs = set()
                for part in a.split(","):
                    if "-" in part: x, y = part.split("-"); cs |= set(range(int(x), int(y) + 1))
                    elif part.strip(): cs.add(int(part))
                if cs and cs <= ktset and len(cs) <= 2: return "KT_NUMA_worker"
            except Exception: pass
        if comm.startswith("sglang::sched"): return "scheduler_main_or_child"
        if "python" in comm or "tokenizer" in comm.lower(): return "python/tokenizer"
        return "unknown"
    out = collections.defaultdict(lambda: {"tids": 0, "ticks": 0, "comms": collections.Counter()}); W = (fg, ld); n_in = 0
    for (pid, tid), seq in by.items():
        seq.sort(); inside = [s for s in seq if W[0] <= s[0] <= W[1]]
        if len(inside) < 2: continue
        d = inside[-1][3] - inside[0][3]; span = (inside[-1][0] - inside[0][0]) / 1e9
        r = role(inside[0][1], tid, None); out[r]["tids"] += 1; out[r]["ticks"] += d; out[r]["comms"][inside[0][1]] += 1; n_in += 1
        out[r].setdefault("span_s", span)
    res = {"window_s": (ld - fg) / 1e9, "clk_tck": hz, "tids_with_samples": n_in, "by_role": {}}
    for r, v in out.items(): res["by_role"][r] = {"tids": v["tids"], "cpu_seconds": v["ticks"] / hz, "cpu_equivalents": (v["ticks"] / hz) / v.get("span_s", 1.0) if v.get("span_s") else None, "comms": dict(v["comms"].most_common(5)), "note": "샘플 구간 [첫 표본, 마지막 표본] 기준 (창 경계 ±1 s)"}
    os.makedirs(f"{sd}/v2", exist_ok=True); json.dump(res, open(f"{sd}/v2/cpu_time_by_role.json", "w"), indent=1, ensure_ascii=False); print(json.dumps({k: (v["tids"], round(v["cpu_equivalents"] or 0, 1)) for k, v in res["by_role"].items()}))


if __name__ == "__main__": main(sys.argv[1])
