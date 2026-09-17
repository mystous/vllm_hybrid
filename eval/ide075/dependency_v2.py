#!/usr/bin/env python3
"""IDE_075 A02/A05 — producer/consumer v2 지표 (지시서 §5). IDE_074 P1 세션(레거시 kt_evt v1: stamp 기반 t_def_start/end) 과
IDE_075 세션(v2: task ring + fwd entry/exit) 둘 다 처리. 기존 파일은 덮어쓰지 않고 <session>/v2/ 에 쓴다.
정의(§5.1~5.2): t_hot_ready(c)=HtoD 직전 커널 종료, t_pub(c)=done store bracket(v2: [t_done_store_b,t_done_store_a]; v1: t_done_set 단일점), t_h2d_start/end(c), t_combine_start/end(c),
  producer p = 층 L−1 의 deferred 작업(같은 replay) — 현재 경로의 후보 매핑(VALIDATED_HEURISTIC_MAPPING: (slot,epoch)↔(layer,replay) 계수 일치 + go≥dtoh_end 순서 + HtoD 직후 결합).
  v2: go_to_enqueue = t_def_enq_a−t_go, enqueue_to_start = t_exec_start(def task)−t_def_enq_a, deferred_service_span = t_fwd_exit−t_fwd_entry (v1: t_def_end−t_def_start),
      predecessor_exec_union = Wq=[enq_a, exec_start) 안의 선행 task 실행 구간 union (v2 만; v1 은 NOT_COLLECTED), queue_gap_unattributed = |Wq| − 그 union.
cohort: no_cold_consumed(L==0), cold_path_empty(prev n_cold_ids==0 또는 def_skipped), cold_present_nonempty, unknown_path_or_mapping.
민감도: cold_pub_delta 의 late 비율을 오차 임계 0/1/2/5 µs 에서 각각 보고 (임계를 바꿔 통과 선택하지 않음; 전부 기록).
항등식: residual = (pub−hot) − (Q+C+D−B) (같은 timestamp 사용 → 0 이어야 함; 단위 확인용).
gpu_idle_inside_gap: G_c=[hot_ready,h2d_start) ∩ TP0 전체 op union 의 여집합.
사용: dependency_v2.py <session_dir> [--v2]"""
import gzip, json, os, sys, csv, collections, statistics as st, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ide074"))
from gpu_layer_timeline import main as layer_timeline
from gpu_union import UnionIndex
HOME = os.path.expanduser("~")
KT_HOSTS = {"ide074": f"{HOME}/.cache/huggingface/kt/ide074", "ide075": f"{HOME}/.cache/huggingface/kt/ide075"}
OPS = ("kernel", "gpu_memcpy", "gpu_memset")


def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None
def summ(v): return {"n": len(v), "p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99), "min": min(v), "max": max(v), "mean": st.mean(v)} if v else {"n": 0}


def load_trace_union(tf):
    j = json.load(gzip.open(tf)); base = j.get("baseTimeNanoseconds", 0) / 1e3
    return UnionIndex([(base + e["ts"], base + e["ts"] + (e.get("dur") or 0)) for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") in OPS])


def main(sd, v2=False):
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]; camp = "ide075" if "IDE_075" in sd else "ide074"; KT = KT_HOSTS[camp]
    out = f"{sd}/v2"; os.makedirs(out, exist_ok=True)
    prof = next((c for c in m["observer"]["collectors"] if c["collector_id"] == "torch_profiler"), None)
    tf0 = next((f"{KT}/{boot}/profiles/{n}" for n, _ in (prof or {}).get("trace_files", []) if "TP-0" in n), None)
    if not tf0: print("no TP-0 trace"); return
    gl_path = f"{sd}/gpu_layer_timeline_TP0.csv.gz"
    if not os.path.exists(gl_path): layer_timeline(tf0, gl_path)
    gl = [r for r in csv.DictReader(gzip.open(gl_path, "rt")) if r.get("t_htod_start")]
    U = load_trace_union(tf0)
    evt = f"{KT}/{boot}/kt_evt.csv"; mp = f"{KT}/{boot}/kt_evt.csv.map"; tasks_p = f"{KT}/{boot}/kt_evt.csv.tasks"
    t0, t1 = m["t_start"]["epoch"] * 1e9, m["drain"]["t_drain_end"]["epoch"] * 1e9
    ev = [r for r in csv.DictReader(l for l in open(evt) if not l.startswith("#")) if t0 <= int(r["t_go"]) <= t1]
    # slot 맵 v2: effective_from = 기록 ns; capturing==1 행만 graph 슬롯
    smap = collections.defaultdict(list)
    for l in open(mp):
        p = l.strip().split(",")
        if len(p) >= 5: smap[int(p[0])].append((int(p[4]), int(p[1]), int(p[2]), int(p[3])))
    for s in smap: smap[s].sort()
    def map_at(slot, t_go):
        rows_ = smap.get(slot, []); i = bisect.bisect_right([x[0] for x in rows_], t_go) - 1
        return rows_[i] if i >= 0 else None
    cap = {}
    for s, rows_ in smap.items():
        for ns, L, rows, capn in rows_:
            if capn == 1: cap[(L, rows)] = s
    tasks = {}
    if v2 and os.path.exists(tasks_p):
        for r in csv.DictReader(l for l in open(tasks_p) if not l.startswith("#")):
            if r.get("t_exec_start"): tasks[int(r["seq"])] = r
    ev_by_slot = collections.defaultdict(list)
    for r in ev: ev_by_slot[int(r["slot"])].append(r)
    for s in ev_by_slot: ev_by_slot[s].sort(key=lambda r: int(r["t_go"]))
    gl_by = collections.defaultdict(list)
    for r in gl: gl_by[(int(r["graph_id"]), int(r["layer_ordinal"]))].append(r)
    for k in gl_by: gl_by[k].sort(key=lambda r: float(r["t_htod_start"]))
    cov = collections.Counter(); recmap = {}
    for (g, L), rows_ in gl_by.items():
        rows_n = int(rows_[0]["htod_bytes"]) // 12288; s = cap.get((L, rows_n)); cov["gpu_layer_rows_total"] += len(rows_)
        if s is None: cov["unmatched_no_slot_map"] += len(rows_); continue
        evs = ev_by_slot.get(s, [])
        if len(evs) != len(rows_): cov["count_mismatch_pairs"] += 1; cov["count_mismatch_rows"] += abs(len(evs) - len(rows_)); continue   # 계수 불일치 → 이 (graph,layer) 전체 unresolved (zip 이동 금지)
        for k, (a, b) in enumerate(zip(rows_, evs)):
            mm = map_at(s, int(b["t_go"]))
            if not mm or mm[1] != L or mm[2] != rows_n: cov["map_at_go_mismatch"] += 1; continue
            recmap[(g, k, L)] = (a, b, s)
    rows = []
    def f(b, k): v = int(b.get(k) or 0); return v / 1e3 if v else None
    for (g, k, L), (a, b, s) in sorted(recmap.items()):
        hot = float(a["t_prev_op_end"]); h2d_s = float(a["t_htod_start"]); h2d_e = float(a["t_htod_end"]); cmb_s = float(a["t_combine_start"]) if a.get("t_combine_start") else None; cmb_e = float(a["t_combine_end"]) if a.get("t_combine_end") else None
        dtoh_e = float(a["t_dtoh_last_end"]); go = int(b["t_go"]) / 1e3
        pub_b = f(b, "t_done_store_b") if v2 else f(b, "t_done_set"); pub_a = f(b, "t_done_store_a") if v2 else pub_b
        prev = recmap.get((g, k, L - 1)); pb = prev[1] if prev else None
        cohort = "no_cold_consumed" if L == 0 else ("unknown_path_or_mapping" if pb is None else ("cold_path_empty" if (int(pb["n_cold_ids"]) == 0 or int(pb["def_skipped"])) else "cold_present_nonempty"))
        row = {"graph_id": g, "replay": k, "consumer_layer": L, "producer_layer": L - 1 if pb is not None else None, "slot": s, "epoch": b["epoch"], "step_name": a["step_name"], "capture_rows": int(a["htod_bytes"]) // 12288, "cpu_qlen": b["qlen"], "path_cohort": cohort,
               "t_go_us": go, "t_dtoh_last_end_us": dtoh_e, "go_minus_dtoh_end_us": go - dtoh_e, "t_hot_ready_us": hot, "t_pub_before_us": pub_b, "t_pub_after_us": pub_a, "t_h2d_start_us": h2d_s, "t_h2d_end_us": h2d_e, "t_combine_start_us": cmb_s,
               "gpu_pre_h2d_gap_us": h2d_s - hot, "h2d_duration_us": h2d_e - h2d_s, "gpu_post_h2d_gap_us": (cmb_s - h2d_e) if cmb_s else None, "cold_gpu_ready_us": h2d_e, "cold_gpu_ready_lateness_us": max(0.0, h2d_e - hot), "combine_schedule_gap_us": (cmb_s - max(hot, h2d_e)) if cmb_s else None,
               "mapping_method": "VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)", "clock_validity": "OK" if go - dtoh_e >= 0 else ("CLOCK_INDETERMINATE" if go - dtoh_e >= -5 else "CLOCK_ORDER_VIOLATION")}
        busy, _ = U.busy((hot, h2d_s)); row["gpu_ops_inside_gap_us"] = busy; row["gpu_idle_inside_gap_us"] = (h2d_s - hot) - busy
        if pub_b is not None:
            row["cold_pub_delta_us"] = pub_b - hot; row["cold_pub_delta_low_us"] = pub_b - hot; row["cold_pub_delta_high_us"] = (pub_a - hot) if pub_a else None
            row["cold_pub_lateness_us"] = max(0.0, pub_b - hot); row["pub_to_h2d_start_us"] = h2d_s - pub_b; row["hot_gpu_ready_lateness_us"] = max(0.0, hot - h2d_e)
        if pb is not None:
            pg = int(pb["t_go"]) / 1e3
            if v2:
                enq_a = f(pb, "t_def_enq_a"); fe = f(pb, "t_fwd_entry"); fx = f(pb, "t_fwd_exit"); seq = int(pb.get("def_task_seq") or 0); tk = tasks.get(seq)
                ex_s = int(tk["t_exec_start"]) / 1e3 if tk else None; ex_e = int(tk["t_exec_end"]) / 1e3 if tk else None
                row.update({"producer_def_task_seq": seq, "go_to_enqueue_us": (enq_a - pg) if enq_a else None, "enqueue_to_start_us": (ex_s - enq_a) if (ex_s and enq_a) else None, "deferred_service_span_us": (fx - fe) if (fe and fx) else None, "task_exec_span_us": (ex_e - ex_s) if (ex_s and ex_e) else None, "fwd_entry_after_exec_start_us": (fe - ex_s) if (fe and ex_s) else None})
                # 선행 task 실행 구간 union ∩ Wq
                if enq_a and ex_s and tasks:
                    pe = []; q = seq - 1
                    while q in tasks and int(tasks[q]["t_exec_end"]) / 1e3 > enq_a and q > seq - 64:
                        t = tasks[q]; pe.append((max(enq_a, int(t["t_exec_start"]) / 1e3), min(ex_s, int(t["t_exec_end"]) / 1e3), t["kind"])); q -= 1
                    pu = sum(max(0.0, b_ - a_) for a_, b_, _ in pe); row["predecessor_exec_union_us"] = pu; row["queue_gap_unattributed_us"] = (ex_s - enq_a) - pu; row["predecessor_kinds"] = "|".join(kk for _, _, kk in pe)
                    row["predecessor_deferred_exec_us"] = sum(max(0.0, b_ - a_) for a_, b_, kk in pe if kk == "3")
                ds, de = fe, fx
            else:
                ds, de = f(pb, "t_def_start"), f(pb, "t_def_end")
                row.update({"go_to_enqueue_us": (f(pb, "t_def_enq") - pg) if f(pb, "t_def_enq") else None, "enqueue_to_start_us": (ds - f(pb, "t_def_enq")) if (ds and f(pb, "t_def_enq")) else None, "deferred_service_span_us": (de - ds) if (ds and de) else None, "predecessor_exec_union_us": None, "queue_gap_unattributed_us": None, "legacy_note": "v1 stamp task 기반; predecessor union NOT_COLLECTED"})
            n0e, n1e, n0s, n1s = f(pb, "t_numa0_end"), f(pb, "t_numa1_end"), f(pb, "t_numa0_start"), f(pb, "t_numa1_start")
            if n0e and n1e: row["numa_completion_skew_us"] = abs(n0e - n1e); row["numa_critical_done_us"] = max(n0e, n1e); row["numa0_span_us"] = n0e - n0s; row["numa1_span_us"] = n1e - n1s
            if de and pub_b is not None: row["producer_end_to_pub_us"] = pub_b - de; row["producer_end_to_gpu_ready_us"] = h2d_e - de
            row["overlap_budget_us"] = hot - pg; row["n_cold_assignments"] = int(pb["n_cold_ids"]); row["producer_qlen"] = pb["qlen"]
            if v2: row.update({"n_unique_cold_experts_numa0": pb.get("act0"), "n_unique_cold_experts_numa1": pb.get("act1"), "max_rows_numa0": pb.get("maxrows0"), "max_rows_numa1": pb.get("maxrows1"), "sum_rows_numa0": pb.get("sumrows0"), "sum_rows_numa1": pb.get("sumrows1"), "n_amx_numa0": pb.get("namx0"), "n_amx_numa1": pb.get("namx1"), "n_avx_numa0": pb.get("navx0"), "n_avx_numa1": pb.get("navx1"), **{f"stage_numa0_{k_}": pb.get(f"s0_{k_}") for k_ in ("prepare", "cpy_input", "q_input", "up_gate", "act", "q_down", "down", "weight", "total")}, **{f"stage_numa1_{k_}": pb.get(f"s1_{k_}") for k_ in ("prepare", "cpy_input", "q_input", "up_gate", "act", "q_down", "down", "weight", "total")}})
            # 항등식 residual (같은 timestamp): Q = ds−go(p), C = de−ds, D = pub−de, B = hot−go(p)
            if ds and de and pub_b is not None: row["identity_residual_us"] = (pub_b - hot) - ((ds - pg) + (de - ds) + (pub_b - de) - (hot - pg))
        rows.append(row)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with gzip.open(f"{out}/producer_consumer_metrics_v2.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # 요약: graph × cohort
    S = {}; sens = {}
    for g in sorted(set(r["graph_id"] for r in rows)):
        for coh in ("cold_present_nonempty", "cold_path_empty", "no_cold_consumed", "unknown_path_or_mapping"):
            rs = [r for r in rows if r["graph_id"] == g and r["path_cohort"] == coh]
            if not rs: continue
            key = f"{g}:{coh}"; S[key] = {"step_names": sorted(set(r["step_name"] for r in rs if r["step_name"]))[:2], "n": len(rs)}
            for kk in ("cold_pub_delta_us", "cold_pub_lateness_us", "gpu_pre_h2d_gap_us", "h2d_duration_us", "gpu_post_h2d_gap_us", "cold_gpu_ready_lateness_us", "combine_schedule_gap_us", "gpu_idle_inside_gap_us", "gpu_ops_inside_gap_us", "go_to_enqueue_us", "enqueue_to_start_us", "deferred_service_span_us", "task_exec_span_us", "predecessor_exec_union_us", "predecessor_deferred_exec_us", "queue_gap_unattributed_us", "numa_completion_skew_us", "numa0_span_us", "numa1_span_us", "producer_end_to_pub_us", "producer_end_to_gpu_ready_us", "pub_to_h2d_start_us", "overlap_budget_us", "identity_residual_us", "n_cold_assignments", "go_minus_dtoh_end_us"):
                v = [r[kk] for r in rs if r.get(kk) is not None]
                if v: S[key][kk] = summ(v)
            if coh == "cold_present_nonempty":
                d = [r["cold_pub_delta_us"] for r in rs if r.get("cold_pub_delta_us") is not None]
                sens[key] = {f"late_share_thr_{t}us": (sum(1 for x in d if x > t) / len(d)) if d else None for t in (0, 1, 2, 5)}
                sens[key]["clock_indeterminate_share_5us"] = (sum(1 for x in d if -5 <= x <= 5) / len(d)) if d else None; sens[key]["n"] = len(d)
                sens[key]["cold_gpu_ready_late_share"] = (sum(1 for r in rs if r["cold_gpu_ready_lateness_us"] > 0) / len(rs))
    res = {"session": sd, "v2": v2, "coverage": dict(cov), "matched_rows": len(rows), "by_graph_cohort": S, "sensitivity": sens, "clock_validity_counts": dict(collections.Counter(r["clock_validity"] for r in rows))}
    json.dump(res, open(f"{out}/dependency_summary_v2.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{out}/missing_coverage_v2.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["item", "count"]); [w.writerow([k, v]) for k, v in sorted(cov.items())]; w.writerow(["matched_rows", len(rows)]); [w.writerow([f"clock:{k}", v]) for k, v in res["clock_validity_counts"].items()]
    for key, v in S.items():
        if "cold_present_nonempty" in key: print(key, v["step_names"], "n", v["n"], "pub_delta p50", (v.get("cold_pub_delta_us") or {}).get("p50"), "pre_h2d p50", (v.get("gpu_pre_h2d_gap_us") or {}).get("p50"), "idle_in_gap p50", (v.get("gpu_idle_inside_gap_us") or {}).get("p50"), "service p50", (v.get("deferred_service_span_us") or {}).get("p50"), "enq→start p50", (v.get("enqueue_to_start_us") or {}).get("p50"), "pred_union p50", (v.get("predecessor_exec_union_us") or {}).get("p50"), "unattributed p50", (v.get("queue_gap_unattributed_us") or {}).get("p50"), "residual max", (v.get("identity_residual_us") or {}).get("max"))
    print("coverage", dict(cov), "clock", res["clock_validity_counts"], "sens", {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sens.items()})


if __name__ == "__main__": main(sys.argv[1], "--v2" in sys.argv)
