#!/usr/bin/env python3
"""IDE_074 M7 — P1 세션의 CPU 이벤트(kt_evt.csv, .map) 와 GPU 층 타임라인(TP-0 트레이스) 을 (slot,epoch)↔(layer,replay) 로 연결해
§10.1/§10.2 지표를 직접 계산한다. 필수 이벤트·clock 정렬이 없으면 계산하지 않고 missing_coverage 에 계수한다.
실행 구조(PLAN_RESOLVED §2): 층 L 의 GPU 결합은 done[slot_L] 을 기다리고, done[slot_L] 은 deferred(L−1) 작업 완료 직후 set 됨.
  t_hot_ready(L)   = 층 L hot MoE 마지막 커널 종료 (GPU)
  t_cold_pub(L)    = rec(slot_L).t_done_set (CPU, done 플래그 게시)
  t_htod_start(L)  = wait 해제 후 첫 GPU 노드 (cold 출력 HtoD)  ← cold 를 GPU 가 실제 소비 가능해지는 시각의 상한
  cold_ready_lateness = max(0, t_cold_pub − t_hot_ready)   (other_ready ≤ hot_ready 이므로 max(hot, other)=hot)
  exposed_wait        = t_htod_start − t_hot_ready          (GPU 유휴로 노출된 대기, 하한 ≈ memop+memcpy 발사 지연)
  publish_to_release  = t_htod_start − t_cold_pub           (음수면 순서 위반 → clock/매핑 오류로 계수)
  cold 경로(층 L 이 소비한 deferred(L−1)): submit_to_start = t_def_start − t_go, cpu_compute_span = t_def_end − t_def_start,
    numa_skew = |numa0_end − numa1_end|, publish_delay = t_cold_pub(L) − t_def_end(L−1), cpu_to_gpu_ready = t_htod_end(L) − t_cold_pub(L),
    cold_budget = t_hot_ready(L) − t_go(L−1)  (CPU 가 hot 과 겹쳐 쓸 수 있던 시간)
사용: dependency_metrics.py <session_dir>  →  <session_dir>/dependency_metrics.csv.gz, dependency_summary.json, missing_coverage.csv"""
import gzip, json, os, sys, csv, collections, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_layer_timeline import main as layer_timeline
HOME = os.path.expanduser("~"); KT_HOST = f"{HOME}/.cache/huggingface/kt/ide074"


def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None
def summ(v): return {"n": len(v), "p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99), "max": max(v) if v else None, "min": min(v) if v else None, "mean": st.mean(v) if v else None} if v else {"n": 0}


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]
    prof = next((c for c in m["observer"]["collectors"] if c["collector_id"] == "torch_profiler"), None)
    tf0 = next((f"{KT_HOST}/{boot}/profiles/{n}" for n, _ in (prof or {}).get("trace_files", []) if "TP-0" in n), None)
    if not tf0: print("no TP-0 trace"); return
    gl_path = f"{sd}/gpu_layer_timeline_TP0.csv.gz"
    if not os.path.exists(gl_path): layer_timeline(tf0, gl_path)
    gl = [r for r in csv.DictReader(gzip.open(gl_path, "rt")) if r.get("t_htod_start")]
    evt = f"{KT_HOST}/{boot}/kt_evt.csv"; mp = f"{KT_HOST}/{boot}/kt_evt.csv.map"
    ev = [r for r in csv.DictReader(l for l in open(evt) if not l.startswith("#"))]
    t0, t1 = m["t_start"]["epoch"] * 1e9, m["drain"]["t_drain_end"]["epoch"] * 1e9
    ev = [r for r in ev if t0 <= int(r["t_go"]) <= t1]
    cap = {}
    for l in open(mp):
        p = l.strip().split(",")
        if len(p) >= 5 and int(p[3]) == 1: cap[(int(p[1]), int(p[2]))] = int(p[0])
    ev_by_slot = collections.defaultdict(list)
    for r in ev: ev_by_slot[int(r["slot"])].append(r)
    for s in ev_by_slot: ev_by_slot[s].sort(key=lambda r: int(r["t_go"]))
    gl_by = collections.defaultdict(list)
    for r in gl: gl_by[(int(r["graph_id"]), int(r["layer_ordinal"]))].append(r)
    for k in gl_by: gl_by[k].sort(key=lambda r: float(r["t_htod_start"]))
    cov = collections.Counter(); rows = []; viol = collections.Counter()
    # (graph, layer) → slot, 그리고 replay k ↔ epoch k
    recmap = {}
    for (g, L), rows_ in gl_by.items():
        rows_n = int(rows_[0]["htod_bytes"]) // 12288; s = cap.get((L, rows_n))
        cov["gpu_layer_rows_total"] += len(rows_)
        if s is None: cov["unmatched_no_slot_map"] += len(rows_); continue
        evs = ev_by_slot.get(s, [])
        if len(evs) != len(rows_): cov["count_mismatch_pairs"] += 1; cov["count_mismatch_rows"] += abs(len(evs) - len(rows_))
        for k, (a, b) in enumerate(zip(rows_, evs)): recmap[(g, k, L)] = (a, b, s)
    for (g, k, L), (a, b, s) in sorted(recmap.items()):
        hot = float(a["t_prev_op_end"]); pub = int(b["t_done_set"]) / 1e3 if int(b["t_done_set"]) else None; htod_s = float(a["t_htod_start"]); htod_e = float(a["t_htod_end"]); dtoh_e = float(a["t_dtoh_last_end"]); go = int(b["t_go"]) / 1e3
        prevrec = recmap.get((g, k, L - 1)); pb = prevrec[1] if prevrec else None
        row = {"graph_id": g, "replay": k, "layer": L, "slot": s, "epoch": b["epoch"], "step_name": a["step_name"], "qlen": b["qlen"], "n_cold_ids_deferred_from_this_layer": b["n_cold_ids"], "def_present": b["def_present"], "def_skipped": b["def_skipped"],
               "t_dtoh_last_end_us": dtoh_e, "t_go_us": go, "go_minus_dtoh_end_us": go - dtoh_e, "t_hot_ready_us": hot, "t_cold_pub_us": pub, "t_htod_start_us": htod_s, "t_htod_end_us": htod_e, "t_combine_start_us": float(a["t_combine_start"]) if a.get("t_combine_start") else None,
               "hot_kernel_sum_us": float(a["hot_kernel_sum_us"]), "validity": []}
        if go - dtoh_e < -5.0: viol["t_go_before_dtoh_end_beyond_5us"] += 1; row["validity"].append("CLOCK_ORDER_VIOLATION(go<dtoh_end-5us)")
        elif go - dtoh_e < 0: viol["t_go_before_dtoh_end_within_5us(jitter)"] += 1   # 정렬 지터로 분류, 원값 보존, 행은 유효
        if pub is None: row["validity"].append("PARTIAL_DEPENDENCY_MEASUREMENT(no_done_set)"); cov["no_done_set"] += 1
        else:
            row["cold_ready_lateness_us"] = max(0.0, pub - hot); row["cold_pub_minus_hot_ready_us"] = pub - hot; row["publish_to_release_us"] = htod_s - pub
            if htod_s - pub < -5.0: viol["htod_before_done_set_beyond_5us"] += 1; row["validity"].append("ORDER_VIOLATION(htod<done_set-5us)")
            elif htod_s - pub < 0: viol["htod_before_done_set_within_5us(jitter)"] += 1
            row["cpu_to_gpu_ready_us"] = htod_e - pub
        row["exposed_wait_us"] = htod_s - hot
        if pb is not None and int(pb["t_def_start"]) and int(pb["t_def_end"]):
            ds, de, pg = int(pb["t_def_start"]) / 1e3, int(pb["t_def_end"]) / 1e3, int(pb["t_go"]) / 1e3
            row.update({"prev_layer_t_go_us": pg, "prev_def_enq_us": int(pb["t_def_enq"]) / 1e3 if int(pb["t_def_enq"]) else None, "prev_def_start_us": ds, "prev_def_end_us": de, "submit_to_start_us": ds - pg, "cpu_compute_span_us": de - ds, "cold_budget_us": hot - pg,
                        "prev_n_cold_ids": pb["n_cold_ids"], "prev_qlen": pb["qlen"]})
            n0s, n0e, n1s, n1e = (int(pb[x]) / 1e3 if int(pb[x]) else None for x in ("t_numa0_start", "t_numa0_end", "t_numa1_start", "t_numa1_end"))
            if n0e and n1e: row["numa_completion_skew_us"] = abs(n0e - n1e); row["numa0_span_us"] = n0e - n0s; row["numa1_span_us"] = n1e - n1s; row["numa_parent_done_us"] = max(n0e, n1e); row["numa_fork_wait_us"] = min(n0s, n1s) - ds
            if pub is not None: row["publish_delay_us"] = pub - de
        else:
            row["validity"].append("NO_PREV_DEFERRED_RECORD" if L > 0 else "LAYER0(no cold consumed)"); cov["no_prev_deferred"] += 1
        row["validity"] = ";".join(row["validity"]) or "OK"; rows.append(row)
    cov["matched_rows"] = len(rows)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with gzip.open(f"{sd}/dependency_metrics.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(f"{sd}/missing_coverage.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "count"]); [w.writerow([k, v]) for k, v in sorted(cov.items())]; [w.writerow([f"violation:{k}", v]) for k, v in viol.items()]
        w.writerow(["cpu_records_in_session", len(ev)]); w.writerow(["gpu_layer_rows", len(gl)]); w.writerow(["slots_mapped_capturing", len(cap)])
    # 요약 (graph/step 별, 유효 행만)
    S = {}
    for g in sorted(set(r["graph_id"] for r in rows)):
        rs = [r for r in rows if r["graph_id"] == g and r["validity"] in ("OK", "LAYER0(no cold consumed)")]
        ok = [r for r in rs if r["validity"] == "OK"]
        S[str(g)] = {"step_names": sorted(set(r["step_name"] for r in rs if r["step_name"]))[:3], "rows_valid": len(rs), "rows_with_prev": len(ok),
                     "cold_ready_lateness_us": summ([r["cold_ready_lateness_us"] for r in rs if "cold_ready_lateness_us" in r]), "cold_pub_minus_hot_ready_us": summ([r["cold_pub_minus_hot_ready_us"] for r in rs if "cold_pub_minus_hot_ready_us" in r]),
                     "exposed_wait_us": summ([r["exposed_wait_us"] for r in rs]), "publish_to_release_us": summ([r["publish_to_release_us"] for r in rs if "publish_to_release_us" in r]), "go_minus_dtoh_end_us": summ([r["go_minus_dtoh_end_us"] for r in rs]),
                     "submit_to_start_us": summ([r["submit_to_start_us"] for r in ok]), "cpu_compute_span_us": summ([r["cpu_compute_span_us"] for r in ok]), "cold_budget_us": summ([r["cold_budget_us"] for r in ok]),
                     "numa_completion_skew_us": summ([r["numa_completion_skew_us"] for r in ok if "numa_completion_skew_us" in r]), "numa0_span_us": summ([r["numa0_span_us"] for r in ok if "numa0_span_us" in r]), "numa1_span_us": summ([r["numa1_span_us"] for r in ok if "numa1_span_us" in r]),
                     "publish_delay_us": summ([r["publish_delay_us"] for r in ok if "publish_delay_us" in r]), "cpu_to_gpu_ready_us": summ([r["cpu_to_gpu_ready_us"] for r in rs if "cpu_to_gpu_ready_us" in r]), "hot_kernel_sum_us": summ([r["hot_kernel_sum_us"] for r in rs]),
                     "share_cold_later_than_hot": (sum(1 for r in rs if r.get("cold_pub_minus_hot_ready_us", 0) > 0) / len(rs)) if rs else None,
                     "prev_n_cold_ids": summ([int(r["prev_n_cold_ids"]) for r in ok])}
    json.dump({"session": sd, "coverage": dict(cov), "violations": dict(viol), "by_graph": S}, open(f"{sd}/dependency_summary.json", "w"), indent=1, ensure_ascii=False)
    for g, v in S.items(): print("graph", g, v["step_names"], "rows", v["rows_valid"], "lateness p50", (v["cold_ready_lateness_us"] or {}).get("p50"), "exposed p50", (v["exposed_wait_us"] or {}).get("p50"), "compute p50", (v["cpu_compute_span_us"] or {}).get("p50"), "submit_to_start p50", (v["submit_to_start_us"] or {}).get("p50"), "budget p50", (v["cold_budget_us"] or {}).get("p50"))
    print("coverage", dict(cov), "violations", dict(viol))


if __name__ == "__main__": main(sys.argv[1])
