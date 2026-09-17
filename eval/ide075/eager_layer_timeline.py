#!/usr/bin/env python3
"""IDE_075 §12.6 — prefill/EXTEND(eager, graph id 0) 층 타임라인 + CPU 레코드 연결.
분절: step[EXTEND ...] annotation 구간 안의 eager op 들에서 cold 출력 HtoD(Pinned→Device, bytes = rows×12288, rows≥2) 순번 = 층(0..61) [HEURISTIC_LAYER_ID].
각 층: DtoH 4건(마지막 종료), 직전 커널 종료(=hot_ready), HtoD start/end, 직후 커널(combine). CPU: kt_evt 에서 같은 step 창 안의 eager 슬롯 레코드(qlen==rows)를 t_go 순으로 대응 (계수 일치 필요).
지표(v2 이름): cold_pub_delta, gpu_pre_h2d_gap, deferred_service_span, enqueue_to_start, stage 단계. 산출: <session>/v2/eager_layer_metrics.csv.gz, eager_summary.json"""
import gzip, json, os, sys, csv, collections, statistics as st, bisect
HOME = os.path.expanduser("~"); KT = f"{HOME}/.cache/huggingface/kt/ide075"; OPS = ("kernel", "gpu_memcpy", "gpu_memset")
ROW_BYTES = int(os.environ.get("KT_ROW_BYTES", "12288"))   # hidden×2 (Qwen 6144→12288, GLM 5120→10240)


def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None
def summ(v): return {"n": len(v), "p50": pct(v, .5), "p95": pct(v, .95), "max": max(v), "min": min(v)} if v else {"n": 0}


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]; prof = next(c for c in m["observer"]["collectors"] if c["collector_id"] == "torch_profiler")
    tf = next(f"{KT}/{boot}/profiles/{n}" for n, _ in prof["trace_files"] if "TP-0" in n)
    j = json.load(gzip.open(tf)); base = j.get("baseTimeNanoseconds", 0) / 1e3
    ops = [e for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") in OPS and not e.get("args", {}).get("graph id")]
    for e in ops: e["_s"] = base + e["ts"]; e["_e"] = e["_s"] + (e.get("dur") or 0)
    ops.sort(key=lambda e: e["_s"])
    steps = sorted(((base + e["ts"], base + e["ts"] + e["dur"], e["name"]) for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[EXTEND")))
    ev = [r for r in csv.DictReader(l for l in open(f"{KT}/{boot}/kt_evt.csv") if not l.startswith("#"))]
    t0, t1 = m["t_start"]["epoch"] * 1e9, m["drain"]["t_drain_end"]["epoch"] * 1e9
    ev = [r for r in ev if t0 <= int(r["t_go"]) <= t1 and int(r["qlen"]) > 1]; ev.sort(key=lambda r: int(r["t_go"]))
    tasks = {int(r["seq"]): r for r in csv.DictReader(l for l in open(f"{KT}/{boot}/kt_evt.csv.tasks") if not l.startswith("#")) if r.get("seq", "").isdigit() and r.get("t_exec_start")}
    rows = []; cov = collections.Counter(); si = 0
    ss = [s[0] for s in steps]
    for k, (a, b, name) in enumerate(steps):
        seg = [e for e in ops if a <= e["_s"] <= b]
        h = [i for i, e in enumerate(seg) if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"] and e.get("args", {}).get("bytes", 0) % ROW_BYTES == 0 and e["args"]["bytes"] >= 2 * ROW_BYTES]
        toks = int(name.split("toks=")[1].rstrip("]")) if "toks=" in name else None
        cpu = [r for r in ev if a * 1e3 <= int(r["t_go"]) <= b * 1e3]
        cov["steps"] += 1; cov["gpu_layers"] += len(h); cov["cpu_recs"] += len(cpu)
        if len(h) != len(cpu) or len(h) == 0: cov["step_count_mismatch"] += 1; continue
        for L, hi in enumerate(h):
            lo = h[L - 1] + 1 if L > 0 else 0; sub = seg[lo:hi]; e = seg[hi]; prev = seg[hi - 1] if hi > 0 else None; nxt = seg[hi + 1] if hi + 1 < len(seg) else None
            dtoh = [x for x in sub if x["cat"] == "gpu_memcpy" and "DtoH" in x["name"]]; r = cpu[L]; pr = cpu[L - 1] if L > 0 else None
            hot = prev["_e"] if prev else None; pub = int(r["t_done_store_b"] or 0) / 1e3 if int(r["t_done_store_b"] or 0) else None
            row = {"step_idx": k, "step": name, "toks": toks, "layer": L, "rows": e["args"]["bytes"] // ROW_BYTES, "cpu_qlen": r["qlen"], "slot": r["slot"], "epoch": r["epoch"], "n_cold_ids": r["n_cold_ids"], "t_go_us": int(r["t_go"]) / 1e3, "t_dtoh_last_end_us": dtoh[-1]["_e"] if dtoh else None, "t_hot_ready_us": hot, "t_h2d_start_us": e["_s"], "t_h2d_end_us": e["_e"], "h2d_duration_us": e["_e"] - e["_s"],
                   "gpu_pre_h2d_gap_us": (e["_s"] - hot) if hot else None, "cold_pub_delta_us": (pub - hot) if (pub and hot) else None, "go_minus_dtoh_end_us": (int(r["t_go"]) / 1e3 - dtoh[-1]["_e"]) if dtoh else None, "combine_name": nxt["name"][:40] if nxt else None}
            if pr is not None:
                fe, fx = int(pr["t_fwd_entry"] or 0) / 1e3, int(pr["t_fwd_exit"] or 0) / 1e3; tk = tasks.get(int(pr["def_task_seq"] or 0)); enq = int(pr["t_def_enq_a"] or 0) / 1e3
                row.update({"deferred_service_span_us": (fx - fe) if fe and fx else None, "enqueue_to_start_us": (int(tk["t_exec_start"]) / 1e3 - enq) if (tk and enq) else None, "overlap_budget_us": (hot - int(pr["t_go"]) / 1e3) if hot else None, "producer_n_cold": pr["n_cold_ids"], "producer_act0": pr.get("act0"), "producer_maxrows0": pr.get("maxrows0"), "producer_namx0": pr.get("namx0"), "producer_navx0": pr.get("navx0"), **{f"stage0_{kk}": pr.get(f"s0_{kk}") for kk in ("prepare", "cpy_input", "q_input", "up_gate", "act", "q_down", "down", "weight", "total")}})
            rows.append(row)
    os.makedirs(f"{sd}/v2", exist_ok=True)
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys: keys.append(kk)
    with gzip.open(f"{sd}/v2/eager_layer_metrics.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    S = {}
    for tk_ in sorted(set(r["toks"] for r in rows if r["toks"])):
        rs = [r for r in rows if r["toks"] == tk_ and r["layer"] > 0]
        S[str(tk_)] = {"n": len(rs), **{k_: summ([r[k_] for r in rs if r.get(k_) is not None]) for k_ in ("cold_pub_delta_us", "gpu_pre_h2d_gap_us", "h2d_duration_us", "deferred_service_span_us", "enqueue_to_start_us", "overlap_budget_us", "go_minus_dtoh_end_us")}, "late_share": (sum(1 for r in rs if (r.get("cold_pub_delta_us") or 0) > 0) / len(rs)) if rs else None, "namx0_p50": pct([float(r["producer_namx0"]) for r in rs if r.get("producer_namx0")], .5), "navx0_p50": pct([float(r["producer_navx0"]) for r in rs if r.get("producer_navx0")], .5), "act0_p50": pct([float(r["producer_act0"]) for r in rs if r.get("producer_act0")], .5)}
    json.dump({"session": sd, "coverage": dict(cov), "by_toks": S, "note": "층 = EXTEND step 내 cold HtoD 순번 (HEURISTIC); CPU 대응 = 같은 step 창의 eager 레코드 t_go 순 (계수 일치 step 만)"}, open(f"{sd}/v2/eager_summary.json", "w"), indent=1, ensure_ascii=False)
    print("coverage", dict(cov)); [print(k, v["n"], "pub_delta p50", v["cold_pub_delta_us"].get("p50"), "gap p50", v["gpu_pre_h2d_gap_us"].get("p50"), "service p50", v["deferred_service_span_us"].get("p50"), "budget p50", v["overlap_budget_us"].get("p50"), "late", v["late_share"], "amx/avx", v["namx0_p50"], v["navx0_p50"]) for k, v in S.items()]


if __name__ == "__main__": main(sys.argv[1])
