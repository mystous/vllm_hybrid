#!/usr/bin/env python3
"""IDE_074 M5 — P1 세션(KT_EVT + torch profiler) 계측 정상성 검증 (지시서 §8 표).
입력: <session_dir> (metrics.json 의 observer.trace_files, boot 의 kt_evt.csv/.map)
검사: GPU activity(kernel 이벤트·DtoH/HtoD 존재) / 요청 창(트레이스 첫 step ↔ bench t_start) / correlation((slot,epoch)↔(layer,replay) 매칭 수) /
      clock(t_go ≥ t_dtoh_last_end 위반 수) / 집계(busy ≤ window) / 원시 보존(파싱 가능·#dropped) / 실행 정상성(bench failed==0, healthy_after)
산출: <session_dir>/validation_results.json, 상태명은 PLAN_RESOLVED §6."""
import gzip, json, os, sys, csv, collections, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_layer_timeline import main as layer_timeline
HOME = os.path.expanduser("~"); KT_HOST = f"{HOME}/.cache/huggingface/kt/ide074"


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]; res = {"session": sd, "checks": {}}
    prof = next((c for c in m["observer"]["collectors"] if c["collector_id"] == "torch_profiler"), None)
    traces = [f"{KT_HOST}/{boot}/profiles/{n}" for n, _ in (prof or {}).get("trace_files", [])]
    res["checks"]["execution"] = {"pass": bool(m["summary"]) and m["summary"].get("failed") == 0 and m["rc"] == 0 and m["healthy_after"], "failed": (m["summary"] or {}).get("failed"), "rc": m["rc"], "healthy_after": m["healthy_after"]}
    # GPU activity
    ga = {}
    for tf in traces:
        j = json.load(gzip.open(tf)); c = collections.Counter(e.get("cat") for e in j["traceEvents"] if e.get("ph") == "X")
        names = collections.Counter(e["name"][:22] for e in j["traceEvents"] if e.get("cat") == "gpu_memcpy")
        ga[os.path.basename(tf)] = {"kernel": c.get("kernel", 0), "gpu_memcpy": c.get("gpu_memcpy", 0), "cpu_op": c.get("cpu_op", 0), "memcpy_names": dict(names), "base_ns": j.get("baseTimeNanoseconds")}
        steps = sorted((j["baseTimeNanoseconds"] / 1e3 + e["ts"]) / 1e6 for e in j["traceEvents"] if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step["))
        ga[os.path.basename(tf)]["first_step_epoch"] = steps[0] if steps else None; ga[os.path.basename(tf)]["last_step_epoch"] = steps[-1] if steps else None
    res["checks"]["gpu_activity"] = {"pass": bool(traces) and all(v["kernel"] > 0 for v in ga.values()) and any("Memcpy HtoD (Pinned ->" in v["memcpy_names"] and "Memcpy DtoH (Device ->" in v["memcpy_names"] for v in ga.values()), "per_trace": ga, "status_if_fail": "INVALID_NO_GPU_ACTIVITY"}
    # 요청 창
    tp0 = next((v for k, v in ga.items() if "TP-0" in k), None)
    if tp0 and tp0["first_step_epoch"]:
        res["checks"]["window"] = {"pass": tp0["first_step_epoch"] >= m["t_start"]["epoch"] - 1.0 and tp0["last_step_epoch"] <= m["drain"]["t_drain_end"]["epoch"] + 1.0, "trace_first_step_epoch": tp0["first_step_epoch"], "bench_t_start_epoch": m["t_start"]["epoch"], "bench_t_end_epoch": m["t_end"]["epoch"], "trace_last_step_epoch": tp0["last_step_epoch"], "lead_s(first_step - bench_start)": tp0["first_step_epoch"] - m["t_start"]["epoch"], "status_if_fail": "INVALID_WINDOW"}
    else: res["checks"]["window"] = {"pass": False, "status_if_fail": "INVALID_WINDOW", "reason": "TP-0 트레이스 step annotation 없음"}
    # CPU 이벤트 + correlation + clock
    evt = f"{KT_HOST}/{boot}/kt_evt.csv"; mp = f"{KT_HOST}/{boot}/kt_evt.csv.map"
    ev = [r for r in csv.DictReader(l for l in open(evt) if not l.startswith("#"))] if os.path.exists(evt) else []
    footer = [l.strip() for l in open(evt) if l.startswith("#")] if os.path.exists(evt) else []
    slot_map = {}
    if os.path.exists(mp):
        for l in open(mp):
            p = l.strip().split(",")
            if len(p) >= 5: slot_map.setdefault(int(p[0]), []).append({"layer": int(p[1]), "rows": int(p[2]), "capturing": int(p[3]), "ns": int(p[4])})
    t0, t1 = m["t_start"]["epoch"] * 1e9, m["drain"]["t_drain_end"]["epoch"] * 1e9
    ev_s = [r for r in ev if t0 <= int(r["t_go"]) <= t1]
    res["checks"]["cpu_events"] = {"pass": len(ev_s) > 0 and len(slot_map) > 0, "rows_total": len(ev), "rows_in_session": len(ev_s), "slots_mapped": len(slot_map), "footer": footer[-1:] , "status_if_fail": "PARTIAL_DEPENDENCY_MEASUREMENT"}
    # correlation: (slot→layer,rows) 와 GPU 층 타임라인 (rows 64 graph) 의 (layer, replay) 매칭, clock: t_go ≥ t_dtoh_last_end
    corr = {"pass": False}
    tf0 = next((t for t in traces if "TP-0" in t), None)
    if tf0 and ev_s and slot_map:
        out = f"{sd}/gpu_layer_timeline_TP0.csv.gz"; layer_timeline(tf0, out)
        gl = [r for r in csv.DictReader(gzip.open(out, "rt")) if r.get("t_htod_start")]
        # 캡처 슬롯: capturing==1 인 항목 (rows 별). graph 별 rows 는 htod_bytes/12288
        by_slot = {s: v for s, v in slot_map.items()}
        cap = {}
        for s, v in by_slot.items():
            for x in v:
                if x["capturing"] == 1: cap[(x["layer"], x["rows"])] = s
        # CPU 레코드: slot → 순서대로 epoch
        ev_by_slot = collections.defaultdict(list)
        for r in ev_s: ev_by_slot[int(r["slot"])].append(r)
        matched = 0; viol = 0; lag = []; unmatched = 0; pairs = []
        gl_by = collections.defaultdict(list)
        for r in gl: gl_by[(int(r["graph_id"]), int(r["layer_ordinal"]))].append(r)
        for (g, L), rows_ in gl_by.items():
            rows_.sort(key=lambda r: float(r["t_htod_start"])); rows_n = int(rows_[0]["htod_bytes"]) // 12288
            s = cap.get((L, rows_n))
            if s is None: unmatched += len(rows_); continue
            evs = sorted(ev_by_slot.get(s, []), key=lambda r: int(r["t_go"]))
            # 세션 내 GPU replay k ↔ 세션 내 CPU epoch k (같은 수여야 함)
            if len(evs) != len(rows_): pairs.append({"graph": g, "layer": L, "slot": s, "gpu_replays": len(rows_), "cpu_records": len(evs)})
            for a, b in zip(rows_, evs):
                matched += 1; d = int(b["t_go"]) / 1e3 - float(a["t_dtoh_last_end"]); lag.append(d)
                if d < 0: viol += 1
        lag.sort(); TOL = 5.0   # µs. CPU CLOCK_REALTIME ↔ CUPTI host 시계 정렬 지터 허용 (관측: P1_r3 에서 −2.25 µs 까지, 매핑/계수 불일치 0). 초과 시 매핑·완료 정의 오류로 취급
        viol_gt = sum(1 for d in lag if d < -TOL)
        corr = {"pass": matched > 0 and viol_gt == 0 and unmatched == 0 and not pairs, "matched": matched, "unmatched_gpu_layers": unmatched, "count_mismatch_pairs": pairs[:10], "n_count_mismatch": len(pairs),
                "t_go_minus_dtoh_end_us": {"p50": lag[len(lag) // 2], "min": lag[0], "max": lag[-1]} if lag else None, "clock_violations(t_go<dtoh_end)": viol, "clock_violations_beyond_tolerance": viol_gt, "clock_tolerance_us": TOL,
                "clock_alignment_bound_note": f"음수 최소 {lag[0] if lag else None} µs = 관측된 정렬 오차 상한; ±{TOL} µs 이내 음수는 지터로 분류(원값 보존)", "status_if_fail": "UNMEASURED_CLOCK_ALIGNMENT/PARTIAL_DEPENDENCY_MEASUREMENT"}
    res["checks"]["correlation_clock"] = corr
    res["overall_pass"] = all(v.get("pass") for v in res["checks"].values())
    json.dump(res, open(f"{sd}/validation_results.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v.get("pass") for k, v in res["checks"].items()}), "overall", res["overall_pass"])
    return res


if __name__ == "__main__": main(sys.argv[1])
