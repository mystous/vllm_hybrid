#!/usr/bin/env python3
"""IDE_075 A10/§13.4 — 세션별 항목 유효성 (valid=true 하나로 승인하지 않음).
execution_valid / config_valid(.so SHA·env) / workload_valid(token 합계) / window_valid(first_go·last_deferred_end 존재, 트레이스 창) / clock_valid_by_pair(go≥dtoh_end, 5 µs 민감도) /
mapping_valid(계수 일치·map_at_go) / producer_consumer_valid(prev 레코드 연결율) / counter_scope_valid(perf 대상 PID·pcm 표본) / counter_running_valid(perf -I 행 수·<not counted> 없음) /
sampling_valid(전수: drop 0) / artifact_valid(파일 parse) / observer_distortion_status(OFF 대비, 별도 계산) / lifecycle(패킷당 task 수 = imm+done+def, spsc full/stale 0)
산출: <session>/v2/validation_results_v2.json"""
import gzip, json, os, sys, csv, collections
HOME = os.path.expanduser("~"); KT = f"{HOME}/.cache/huggingface/kt/ide075"


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]; mode = m["mode"]; out = f"{sd}/v2"; os.makedirs(out, exist_ok=True); V = {}
    s = m.get("summary") or {}
    V["execution_valid"] = {"pass": bool(s) and s.get("failed") == 0 and m["rc"] == 0 and m["healthy_after"], "completed": s.get("completed"), "failed": s.get("failed"), "rc": m["rc"]}
    V["workload_valid"] = {"pass": s.get("total_input_tokens") == 65886 and s.get("total_output_tokens") == 16384, "in": s.get("total_input_tokens"), "out": s.get("total_output_tokens"), "expected": "65886/16384 (PROBE128, 지시서 §2.2)"} if (m["workload"] == "M073_QWEN_PROBE128" and m.get("client", "vllm") != "probe") else {"pass": None, "note": "PROBE128 외 또는 probe 클라이언트(입력 토큰 합계 미보고; 비동등 판정으로 대표값 아님)"}
    rq = json.load(open(f"{os.path.dirname(sd.rstrip('/'))}/requested_config.json")); V["config_valid"] = {"pass": rq.get("kt_kernel_so_sha256", "").startswith(("a4e9045a", "d659ca0b", "12926df2")), "so": rq.get("kt_kernel_so_sha256", "")[:16], "mode": rq.get("mode"), "env_kt_evt": (rq.get("env") or {}).get("KT_EVT")}
    we = [json.loads(l) for l in open(f"{sd}/window_events.jsonl")] if os.path.exists(f"{sd}/window_events.jsonl") else []
    names = {e["event_name"] for e in we}
    V["window_valid"] = {"pass": (mode == "OFF") or ({"first_go", "last_related_deferred_end"} <= names), "events": sorted(names), "note": "OFF 는 서버측 창 없음(NOT_COLLECTED)" if mode == "OFF" else None}
    if mode != "OFF":
        evt = f"{KT}/{boot}/kt_evt.csv"; tasks = f"{KT}/{boot}/kt_evt.csv.tasks"
        footer = [l.strip() for l in open(evt) if l.startswith("#")]
        t0, t1 = m["t_start"]["epoch"] * 1e9, m["drain"]["t_drain_end"]["epoch"] * 1e9
        ev = [r for r in csv.DictReader(l for l in open(evt) if not l.startswith("#")) if t0 <= int(r["t_go"]) <= t1]
        n_def = sum(1 for r in ev if int(r["def_present"]) and not int(r["def_skipped"])); n_imm = sum(1 for r in ev if int(r["imm_present"]))
        n_fwd = sum(1 for r in ev if int(r["t_fwd_exit"] or 0) > 0); n_done = sum(1 for r in ev if int(r["t_done_store_a"] or 0) > 0)
        seqs = {int(r["def_task_seq"]) for r in ev if int(r["def_task_seq"] or 0)}; tk = {}
        for r in csv.DictReader(l for l in open(tasks) if not l.startswith("#")):
            if r.get("seq") and r["seq"].isdigit(): tk[int(r["seq"])] = r
        kinds = collections.Counter(tk[q]["kind"] for q in seqs if q in tk)
        V["sampling_valid"] = {"pass": bool(footer) is False or all("dropped=0" in f and "spsc_full=0" in f for f in footer[-1:]), "footer": footer[-1:], "note": "footer 는 서버 종료 시 기록; 세션 중이면 없음", "records_in_session": len(ev)}
        V["lifecycle_valid"] = {"pass": (n_def - n_fwd) <= max(1, n_def // 1000) and n_done == len(ev), "fwd_missing": n_def - n_fwd, "fwd_missing_share": (n_def - n_fwd) / n_def if n_def else None, "criterion": "미기록 ≤ 0.1 % (건수 공개)", "packets": len(ev), "def_tasks_expected": n_def, "fwd_recorded": n_fwd, "done_store_recorded": n_done, "imm_tasks": n_imm, "def_task_seq_found_in_tasks": len(seqs & set(tk)), "def_task_kinds": dict(kinds), "note": "패킷당 업무 task = imm(0/1) + done(1) + def(0/1); 기록 전용 task 0"}
        tsel = [t for q, t in tk.items() if t.get("t_enq_b") and t0 <= int(t["t_enq_b"]) <= t1]
        kc = collections.Counter(t["kind"] for t in tsel)
        V["task_count_valid"] = {"pass": abs(kc.get("3", 0) - n_def) <= 2 and abs(kc.get("2", 0) - len(ev)) <= 2, "tasks_in_window_by_kind": dict(kc), "expected": {"3(deferred)": n_def, "2(done)": len(ev), "1(imm)": n_imm}}
    prof = next((c for c in m["observer"]["collectors"] if c["collector_id"] == "torch_profiler"), None)
    if prof:
        tr = [f"{KT}/{boot}/profiles/{n}" for n, _ in prof.get("trace_files", [])]
        ga = {}
        for tf in tr:
            j = json.load(gzip.open(tf)); c = collections.Counter(e.get("cat") for e in j["traceEvents"] if e.get("ph") == "X"); ga[os.path.basename(tf)[-22:]] = {"kernel": c.get("kernel", 0), "memcpy": c.get("gpu_memcpy", 0)}
        V["gpu_activity_valid"] = {"pass": len(tr) == 4 and all(v["kernel"] > 0 for v in ga.values()), "per_trace": ga}
    dep = f"{out}/dependency_summary_v2.json"
    if os.path.exists(dep):
        d = json.load(open(dep)); cov = d["coverage"]; cl = d["clock_validity_counts"]
        tot = cov.get("gpu_layer_rows_total", 0) or 1; edge_ok = (cov.get("resync_unmatched_gpu", 0) + cov.get("resync_unmatched_cpu", 0)) <= 0.03 * tot and d["matched_rows"] >= 0.97 * tot   # 세션 창 경계의 replay/레코드(층당 ≤1~2건) 만 미대응 허용
        V["mapping_valid"] = {"pass": cov.get("map_at_go_mismatch", 0) == 0 and cov.get("unmatched_no_slot_map", 0) == 0 and (cov.get("count_mismatch_rows", 0) == 0 or edge_ok), "coverage": cov, "matched": d["matched_rows"], "method": "VALIDATED_HEURISTIC_MAPPING" + ("+ANCHOR_RESYNC(창 경계 미대응 ≤3 %)" if cov.get("count_mismatch_rows", 0) else ""), "zip_order_violation_groups": cov.get("zip_order_violation_groups", 0)}
        V["clock_valid_by_pair"] = {"pass": cl.get("CLOCK_ORDER_VIOLATION", 0) == 0, "counts": cl, "sensitivity": d["sensitivity"]}
        big = [v for k, v in d["by_graph_cohort"].items() if k.endswith("cold_present_nonempty")]
        V["producer_consumer_valid"] = {"pass": all((v.get("deferred_service_span_us") or {}).get("n", 0) >= v["n"] - max(1, v["n"] // 1000) for v in big), "linked_over_n": {k: ((v.get("deferred_service_span_us") or {}).get("n", 0), v["n"]) for k, v in d["by_graph_cohort"].items() if k.endswith("cold_present_nonempty")}, "criterion": "연결 ≥ 99.9 % (건수 공개)", "cohorts": {k: v["n"] for k, v in d["by_graph_cohort"].items()}, "residual_max": max(((v.get("identity_residual_us") or {}).get("max") or 0) for v in big) if big else None}
    if mode == "RESOURCE":
        pc = f"{sd}/cpu_counter_intervals.csv"; rows = [l for l in open(pc)] if os.path.exists(pc) else []
        nc = sum(1 for l in rows if "<not counted>" in l or "<not supported>" in l); ni = sum(1 for l in rows if l.strip() and not l.startswith("#"))
        V["counter_running_valid"] = {"pass": ni > 0 and nc == 0, "interval_rows": ni, "not_counted_rows": nc}
        V["counter_scope_valid"] = {"pass": os.path.exists(f"{os.path.dirname(sd.rstrip('/'))}/thread_role_map.csv"), "target": next((c.get("target") for c in m["observer"]["collectors"] if c["collector_id"] == "perf_stat_interval"), None), "note": "process-level -p, 스레드 상속 (perf 문서); TID 별 coverage 는 thread_role_map 와 대조"}
        pcm = f"{sd}/pcm_memory.raw.csv"; V["pcm_artifact_valid"] = {"pass": os.path.exists(pcm) and os.path.getsize(pcm) > 1000, "bytes": os.path.getsize(pcm) if os.path.exists(pcm) else 0}
    V["artifact_valid"] = {"pass": all(os.path.exists(f"{sd}/{f}") for f in ("metrics.json", "requests.jsonl", "cpu_timeseries.csv", "gpu_timeseries.csv", "bench_cmd.sh"))}
    res = {"session": sd, "mode": mode, "checks": V, "all_pass": all(v.get("pass") in (True, None) for v in V.values())}
    json.dump(res, open(f"{out}/validation_results_v2.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v.get("pass") for k, v in V.items()}), "all", res["all_pass"])


if __name__ == "__main__": main(sys.argv[1])
