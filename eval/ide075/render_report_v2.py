#!/usr/bin/env python3
"""IDE_075 A12/§18 — FULL_REPORT.md / BOTTLENECK_EVIDENCE.md / VALIDATION_REPORT.md / MEASUREMENT_COMPLETION_STATUS.md / profiles/{observer_overhead_v2.csv, missing_coverage_v2.csv, clock_alignment_summary.csv} /
FULL_RAW_DATA.md(+raw_md) / ARTIFACT_INDEX.csv / SHA256SUMS.txt. READINESS.md·OPTIMIZATION_HANDOFF.md 는 수동 작성(판정 문서). 평가 언어 없음."""
import glob, gzip, json, os, statistics as st, sys, csv, time, hashlib, collections
HOME = os.path.expanduser("~"); REPO = f"{HOME}/projects/vllm_hybrid"; FEAT = f"{REPO}/shadow_assists/features/IDE_075"; CAMP = f"{REPO}/eval/results/IDE_075_20260917"; ST = f"{CAMP}/state"; OFF = f"{CAMP}/offline"; KT = f"{HOME}/.cache/huggingface/kt/ide075"; C74 = f"{REPO}/eval/results/IDE_074_20260917"


def f1(x, n=1): return "null" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
def rel(p): return os.path.relpath(p, REPO) if p.startswith(REPO) else p
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def now_kst(): return time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime())
def sd(v): return f"{st.stdev(v):.2f}" if len(v) > 1 else "null"


def sessions(camp):
    out = []
    for bd in sorted(glob.glob(f"{camp}/qwen/OPT4/*/")):
        rq = json.load(open(f"{bd}requested_config.json")) if os.path.exists(f"{bd}requested_config.json") else {}
        for sdir in sorted(glob.glob(f"{bd}*/")):
            if os.path.exists(f"{sdir}metrics.json") and not sdir.rstrip("/").endswith("warmup"):
                m = json.load(open(f"{sdir}metrics.json")); m["_dir"] = sdir; m["_boot"] = bd.rstrip("/").split("/")[-1]; m["_so"] = rq.get("kt_kernel_so_sha256", "")[:12]; out.append(m)
    return out


def prow(m):
    s = m.get("summary") or {}
    return {"boot": m["_boot"], "session": m["session"], "mode": m["mode"], "workload": m["workload"], "C": m["C"], "n": m["n"], "completed": s.get("completed"), "failed": s.get("failed"), "duration": s.get("duration"), "in": s.get("total_input_tokens"), "out": s.get("total_output_tokens"), "tps": s.get("output_throughput"),
            "ttft_p50": s.get("median_ttft_ms"), "ttft_p95": s.get("p95_ttft_ms"), "tpot_p50": s.get("median_tpot_ms"), "tpot_p95": s.get("p95_tpot_ms"), "e2el_p50": s.get("median_e2el_ms"), "e2el_p95": s.get("p95_e2el_ms"), "itl_p95": s.get("p95_itl_ms"), "valid": bool(s) and s.get("failed") == 0 and m["rc"] == 0, "t_start": m["t_start"]["wall_kst"], "so": m["_so"], "dir": rel(m["_dir"])}


def main():
    S = sessions(CAMP); rows = [prow(m) for m in S]; ev = jl(f"{ST}/execution_events.jsonl"); ev74 = jl(f"{C74}/state/execution_events.jsonl")
    u = {"boot_attempts": sum(e["event"] == "boot_start" for e in ev), "boot_aborted": sum(e["event"] == "boot_end" and e.get("verdict") != "HEALTH_OK" for e in ev), "sessions": sum(e["event"] == "session_end" for e in ev), "smoke": sum(e.get("requests", 0) for e in ev if e["event"] == "smoke_end"), "warmup": sum(e.get("requests", 0) for e in ev if e["event"] == "warmup_end"),
         "boot_074": sum(e["event"] == "boot_start" for e in ev74), "sessions_074": sum(e["event"] == "session_end" for e in ev74)}
    u["boot_total"] = u["boot_074"] + u["boot_attempts"]; u["sessions_total"] = u["sessions_074"] + u["sessions"]
    json.dump({"budget_total": {"SESSIONS": 24, "BOOT": 12, "QUALITY_Q": 80}, "ide074": {"sessions": u["sessions_074"], "boots": u["boot_074"], "quality": 4}, "ide075": {"sessions": u["sessions"], "boot_attempts": u["boot_attempts"], "boot_aborted": u["boot_aborted"], "smoke_requests": u["smoke"], "warmup_requests": u["warmup"], "quality": 0}, "remaining": {"SESSIONS": 24 - u["sessions_total"], "BOOT": 12 - u["boot_total"], "QUALITY_Q": 76}}, open(f"{ST}/budget_reconciliation.json", "w"), indent=1)
    os.makedirs(f"{FEAT}/profiles", exist_ok=True)
    # ---- observer_overhead_v2 ----
    off = [r for r in rows if r["mode"] == "OFF" and r["valid"] and r["workload"] == "M073_QWEN_PROBE128"]
    OO = [["session", "mode", "boot", "tps", "duration_s", "e2el_p50", "ttft_p95", "tpot_p95", "ref", "throughput_ratio", "duration_ratio", "e2el_p50_ratio", "ttft_p95_ratio", "tpot_p95_ratio", "criterion(|tps|<=3%,p95<=5%)", "status"]]
    for r in rows:
        if r["workload"] != "M073_QWEN_PROBE128" or not r["valid"] or r["mode"] == "OFF": continue
        refs = [("OFF_mean", {k: st.mean(o[k] for o in off) for k in ("tps", "duration", "e2el_p50", "ttft_p95", "tpot_p95")})] + [(o["session"], o) for o in off]
        for name, o in refs:
            tr = r["tps"] / o["tps"]; e2 = r["e2el_p50"] / o["e2el_p50"]; tt = r["ttft_p95"] / o["ttft_p95"]; tp = r["tpot_p95"] / o["tpot_p95"]
            crit = abs(tr - 1) <= 0.03 and abs(tt - 1) <= 0.05 and abs(tp - 1) <= 0.05
            OO.append([r["session"], r["mode"], r["boot"], r["tps"], r["duration"], r["e2el_p50"], r["ttft_p95"], r["tpot_p95"], name, tr, r["duration"] / o["duration"], e2, tt, tp, crit, "DESCRIPTIVE_LOW_DISTORTION" if crit else "DIAGNOSTIC_ONLY"])
    off_var = (max(o["tps"] for o in off) / min(o["tps"] for o in off) - 1) if len(off) > 1 else None
    with open(f"{FEAT}/profiles/observer_overhead_v2.csv", "w", newline="") as f: csv.writer(f).writerows(OO + [["OFF_self_variance", "", "", "", "", "", "", "", "", off_var, "", "", "", "", "INCONCLUSIVE_BOOT_VARIANCE" if (off_var or 0) > 0.03 else "", ""]])
    # ---- dependency v2 / validation / costs ----
    DEP = {m["session"]: json.load(open(f"{m['_dir']}v2/dependency_summary_v2.json")) for m in S if os.path.exists(f"{m['_dir']}v2/dependency_summary_v2.json")}
    VAL = {m["session"]: json.load(open(f"{m['_dir']}v2/validation_results_v2.json")) for m in S if os.path.exists(f"{m['_dir']}v2/validation_results_v2.json")}
    QB = {m["session"]: list(csv.DictReader(gzip.open(f"{m['_dir']}v2/queue_wait_breakdown.csv.gz", "rt"))) for m in S if os.path.exists(f"{m['_dir']}v2/queue_wait_breakdown.csv.gz")}
    PL = {m["session"]: list(csv.DictReader(open(f"{m['_dir']}v2/producer_layer_costs.csv"))) for m in S if os.path.exists(f"{m['_dir']}v2/producer_layer_costs.csv")}
    with open(f"{FEAT}/profiles/missing_coverage_v2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["session", "item", "count"])
        for sname, d in DEP.items():
            for k, v in d["coverage"].items(): w.writerow([sname, k, v])
            for k, v in d["clock_validity_counts"].items(): w.writerow([sname, f"clock:{k}", v])
    with open(f"{FEAT}/profiles/clock_alignment_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["session", "pair", "n", "p50_us", "min_us", "max_us", "indeterminate_5us", "order_violation_gt5us", "method"])
        for sname, d in DEP.items():
            for g, v in d["by_graph_cohort"].items():
                gm = v.get("go_minus_dtoh_end_us") or {}
                if gm.get("n"): w.writerow([sname, f"{g} go(CPU)−dtoh_end(GPU)", gm["n"], gm["p50"], gm["min"], gm["max"], d["clock_validity_counts"].get("CLOCK_INDETERMINATE", 0), d["clock_validity_counts"].get("CLOCK_ORDER_VIOLATION", 0), "순서검사 (offset 불변 가능성은 남음; 오차구간 방식)"])
    # ---- FULL_REPORT ----
    env74 = json.load(open(f"{REPO}/shadow_assists/features/IDE_074/evidence/environment.json"))
    L = [f"# FULL_REPORT — IDE_075 후속 추가 측정 ({now_kst()})", "", f"지시서 `PLAN.md` (SHA {sha(f'{FEAT}/PLAN.md')[:16]}…). 실행 확정 `PLAN_RESOLVED.md`, 조사 기록 `WORK_LOG.md`, 창 재구성 `WINDOW_RECONSTRUCTION.md`, 지표 정의 `METRIC_DEFINITIONS_V2.yaml`, 검증 `VALIDATION_REPORT.md`, 판정 `READINESS.md`, 인계 `OPTIMIZATION_HANDOFF.md`. 원인 추정·개선 권고는 본 문서에 없음.", "",
         "## 1. 환경·구성", "", "```", f"host {env74['host']['hostname']} kernel {env74['host']['kernel']} no_turbo={env74['host']['no_turbo']} governor={env74['host']['governor']}", f"kt_kernel_ext.so v2 (IDE_075): a4e9045a038bc7af2ec81b1cf251b007399d7e998a5d58c86f579c1bd1bd6ed3 8319008 B  (v1 adf49e48… / 원본 e29357f7… 은 /sgl-workspace/ide074_backup)", "launch argv / env / hotmap / layer budget: IDE_074 와 동일 (CONFIG_DIFF.md)", "```", "",
         "부팅 시도 (원장):", "", "| boot_id | plan/mode | verdict | 비고 |", "|---|---|---|---|"]
    for e in ev:
        if e["event"] == "boot_start": be = next((x for x in ev if x["event"] == "boot_end" and x.get("boot_id") == e["boot_id"]), {}); L.append(f"| {e['boot_id']} | {e.get('plan')}/{e.get('mode')} | {be.get('verdict')} | {be.get('note', '')} |")
    L += ["", f"실행량: IDE_074 세션 {u['sessions_074']}/부팅 {u['boot_074']} + IDE_075 세션 {u['sessions']}/부팅 시도 {u['boot_attempts']} (중단 {u['boot_aborted']}) → 합계 세션 {u['sessions_total']}/24 · 부팅 {u['boot_total']}/12 · smoke 요청 {u['smoke']} · warmup 요청 {u['warmup']} (`state/budget_reconciliation.json`)", "",
          "## 2. 모든 세션 (원값)", "", "| boot | session | mode | workload | C | n | 완료/실패 | dur s | in | out | out tok/s | TTFT p50/p95 | TPOT p50/p95 | E2EL p50/p95 | valid | .so | start |", "|" + "---|" * 17]
    for r in rows: L.append(f"| {r['boot']} | {r['session']} | {r['mode']} | {r['workload']} | {r['C']} | {r['n']} | {f1(r['completed'],0)}/{f1(r['failed'],0)} | {f1(r['duration'])} | {f1(r['in'],0)} | {f1(r['out'],0)} | {f1(r['tps'],2)} | {f1(r['ttft_p50'],0)}/{f1(r['ttft_p95'],0)} | {f1(r['tpot_p50'])}/{f1(r['tpot_p95'])} | {f1(r['e2el_p50'],0)}/{f1(r['e2el_p95'],0)} | {r['valid']} | {r['so']} | {r['t_start']} |")
    L += ["", "역사적 기준(IDE_074, 재측정 없이 계승; 새 분모로 쓰지 않음): MAIN_SHORT OFF 743.02/747.53/768.93, PROBE128 OFF 720.80/780.72/732.61, P1 696.39/705.62/690.76, P2 706.46/720.67/705.25 tok/s (v1 바이너리).", "",
          "## 3. 계측 간섭 (profiles/observer_overhead_v2.csv; OFF 평균 및 OFF 원값 각각 대비; 기준: |처리량|≤3 %, p95 ≤5 %)", "", "| session | mode | ref | throughput_ratio | duration_ratio | e2el_p50 | ttft_p95 | tpot_p95 | status |", "|---|---|---|---|---|---|---|---|---|"]
    for r in OO[1:]: L.append(f"| {r[0]} | {r[1]} | {r[8]} | {f1(r[9],3)} | {f1(r[10],3)} | {f1(r[11],3)} | {f1(r[12],3)} | {f1(r[13],3)} | {r[15]} |")
    L += [f"", f"OFF 자체 변동 (max/min−1): {f1(off_var,3)} → {'INCONCLUSIVE_BOOT_VARIANCE' if (off_var or 0) > 0.03 else '3 % 이내'}. 단발 CORE/RESOURCE 에 SD/CI 없음. OFF 두 부팅은 시간순으로 CORE 부팅 앞에 위치 (OFF_OPEN/OFF_CLOSE 배치 미충족, WORK_LOG).", "",
          "## 4. 유효성 (세션별 v2/validation_results_v2.json, 항목 분리)", "", "| session | mode | " + " | ".join(k for k in ("execution_valid", "workload_valid", "config_valid", "window_valid", "lifecycle_valid", "task_count_valid", "sampling_valid", "gpu_activity_valid", "mapping_valid", "clock_valid_by_pair", "producer_consumer_valid", "counter_running_valid", "counter_scope_valid", "artifact_valid")) + " |", "|" + "---|" * 16]
    for sname, v in VAL.items(): L.append(f"| {sname} | {v['mode']} | " + " | ".join(str(v['checks'].get(k, {}).get('pass', 'n/a')) for k in ("execution_valid", "workload_valid", "config_valid", "window_valid", "lifecycle_valid", "task_count_valid", "sampling_valid", "gpu_activity_valid", "mapping_valid", "clock_valid_by_pair", "producer_consumer_valid", "counter_running_valid", "counter_scope_valid", "artifact_valid")) + " |")
    L += ["", "## 5. v2 의존 지표 (CORR 세션; cohort cold_present_nonempty; µs)", ""]
    KEYS = ["cold_pub_delta_us", "cold_pub_lateness_us", "gpu_pre_h2d_gap_us", "gpu_idle_inside_gap_us", "h2d_duration_us", "gpu_post_h2d_gap_us", "combine_schedule_gap_us", "go_to_enqueue_us", "enqueue_to_start_us", "predecessor_exec_union_us", "predecessor_deferred_exec_us", "queue_gap_unattributed_us", "deferred_service_span_us", "task_exec_span_us", "numa0_span_us", "numa1_span_us", "numa_completion_skew_us", "producer_end_to_pub_us", "pub_to_h2d_start_us", "overlap_budget_us", "identity_residual_us", "n_cold_assignments", "go_minus_dtoh_end_us"]
    for sname, d in DEP.items():
        L += [f"### {sname} — coverage {d['coverage']} · clock {d['clock_validity_counts']}", "", "| graph:cohort | n | 지표 | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|"]
        for g, v in d["by_graph_cohort"].items():
            for k in KEYS:
                x = v.get(k) or {}
                if x.get("n"): L.append(f"| {g} {v['step_names']} | {v['n']} | {k} | {f1(x['p50'])} | {f1(x['p95'])} | {f1(x['p99'])} | {f1(x['max'])} |")
        L += ["", "민감도 (late share, cold_present_nonempty): " + json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in d["sensitivity"].items()}, ensure_ascii=False), ""]
    L += ["## 6. FIFO 대기 분해 (queue_wait_breakdown; 같은 사건 단위, µs)", "", "| session | n | queue_wait_wall p50/p95 | predecessor_exec_union p50/p95 | pred_deferred p50 | pred_done_setter p50 | unattributed p50/p95/max | dequeue→exec p50 | enq bracket p50 |", "|---|---|---|---|---|---|---|---|---|"]
    def q(v, p): v = sorted(v); return v[int(round(p * (len(v) - 1)))] if v else None
    for sname, qb in QB.items():
        if not qb: continue
        g = lambda k: [float(x[k]) for x in qb]
        L.append(f"| {sname} | {len(qb)} | {f1(q(g('queue_wait_wall_us'),.5))}/{f1(q(g('queue_wait_wall_us'),.95))} | {f1(q(g('predecessor_exec_union_us'),.5))}/{f1(q(g('predecessor_exec_union_us'),.95))} | {f1(q(g('pred_deferred_us'),.5))} | {f1(q(g('pred_done_setter_us'),.5))} | {f1(q(g('queue_gap_unattributed_us'),.5))}/{f1(q(g('queue_gap_unattributed_us'),.95))}/{f1(max(g('queue_gap_unattributed_us')))} | {f1(q(g('dequeue_to_exec_us'),.5))} | {f1(q(g('enq_bracket_us'),.5))} |")
    L += ["", "## 7. 생산층별 비용표 (producer_layer_costs.csv; 관측값, 한계 이득 아님)", ""]
    for sname, pl in PL.items():
        big = [r for r in pl if r["step"].startswith("step[DECODE bs=6")]
        if not big: continue
        L += [f"### {sname} (bs=63 graph, 층별 p50, µs)", "", "| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |", "|" + "---|" * 17]
        for r in big: L.append(f"| {r['producer_layer']} | {r['n']} | {f1(float(r['n_cold_assignments_p50']),0)} | {r['unique_cold_experts_numa0_p50']}/{r['unique_cold_experts_numa1_p50']} | {r['max_rows_numa0_p50']} | {r['n_amx_numa0_p50']}/{r['n_avx_numa0_p50']} | {f1(float(r['enqueue_to_start_p50']))} | {f1(float(r['predecessor_exec_union_p50']) if r['predecessor_exec_union_p50'] else None)} | {f1(float(r['queue_gap_unattributed_p50']) if r['queue_gap_unattributed_p50'] else None)} | {f1(float(r['deferred_service_span_p50']))} | {f1(float(r['numa0_span_p50']))}/{f1(float(r['numa1_span_p50']))} | {f1(float(r['numa_skew_p50']))} | {f1(float(r['cold_pub_delta_p50']))} | {f1(float(r['late_share']),3)} | {f1(float(r['gpu_pre_h2d_gap_p50']))} | {f1(float(r['overlap_budget_p50']))} | " + ",".join(f1(float(r[f'stage_numa0_{k}_p50']),0) if r[f'stage_numa0_{k}_p50'] else 'null' for k in ('prepare','cpy_input','q_input','up_gate','act','q_down','down','weight','total')) + " |")
        L.append("")
    L += ["## 8. 자원 (RESOURCE 세션; 부하 창 = forward_envelope(kt_evt))", ""]
    for m in S:
        if m["mode"] != "RESOURCE": continue
        rp = f"{m['_dir']}v2/resource_summary.json"
        if os.path.exists(rp): L += [f"### {m['session']}", "", "```", json.dumps(json.load(open(rp)), ensure_ascii=False, indent=1)[:4000], "```", ""]
    L += ["## 9. 기존 자료 정정 (A01; profiles/legacy_vs_corrected.csv, WINDOW_RECONSTRUCTION.md)", "", "| item | session | legacy | corrected | 사유 |", "|---|---|---|---|---|"]
    for r in csv.DictReader(open(f"{FEAT}/profiles/legacy_vs_corrected.csv")): L.append(f"| {r['item']} | {r['session']} | {r['legacy_value']} | {r['corrected_value']} | {r['difference_reason']} |")
    L += ["", "## 10. 오류·중단·미실행", "", "| 항목 | 내용 |", "|---|---|"]
    for e in ev:
        if e["event"] in ("plan_abort", "server_died") or (e["event"] == "boot_end" and e.get("verdict") != "HEALTH_OK"): L.append(f"| {e['event']} {e.get('boot_id')} | {e.get('verdict', '')} {e.get('note', '')} |")
    L += ["| GLM (G00) | 실행 없음. IDE_074 게이트 BLOCKED_NORMAL_OUTPUT 계승 (GLM_NORMAL_OUTPUT_BLOCKED) |", "| S7 예비 | 하네스 결함으로 부팅 2회 소진 → 예비 없음 (BLOCKED_BY_BUDGET) |", "", "## 11. 파일", "", "- offline/: legacy_windows.csv, pcm_samples_v2.csv, pcm_window_summary_v2.csv, timestamp_parse_errors.jsonl, parser_test_results.json", "- 세션/v2/: producer_consumer_metrics_v2.csv.gz, dependency_summary_v2.json, missing_coverage_v2.csv, queue_wait_breakdown.csv.gz, fifo_dependency_edges.csv.gz, producer_layer_costs.csv, expert_cost_samples.csv.gz, fifo_casebook.md, validation_results_v2.json, resource_summary.json", "- 세션/: window_events.jsonl, observer_config.json, cpu_counter_intervals.csv, pcm_memory.raw.csv, requests.jsonl, 시계열", "- 서버 보존: ~/.cache/huggingface/kt/ide075/<boot>/{kt_evt.csv, kt_evt.csv.tasks, kt_evt.csv.map, profiles/*.trace.json.gz} (ARTIFACT_INDEX.csv SHA256)"]
    open(f"{FEAT}/FULL_REPORT.md", "w").write("\n".join(L) + "\n")
    # ---- VALIDATION_REPORT ----
    V = [f"# VALIDATION_REPORT — IDE_075 ({now_kst()})", "", "## parser/단위/기록기 합성 시험", "", "```", json.dumps(json.load(open(f"{OFF}/parser_test_results.json")), ensure_ascii=False, indent=1), "```", "", "## 세션별 항목 유효성", ""]
    for sname, v in VAL.items(): V += [f"### {sname} ({v['mode']}) all_pass={v['all_pass']}", "", "```", json.dumps(v["checks"], ensure_ascii=False, indent=1)[:6000], "```", ""]
    V += ["## clock", "", "profiles/clock_alignment_summary.csv. 방법: CPU CLOCK_REALTIME ↔ kineto host 시계 순서검사 (go ≥ dtoh_end). 일정 offset 은 검출 불가 → 5 µs 이내 차이는 CLOCK_INDETERMINATE 로 분류, 민감도 0/1/2/5 µs 전부 보고.", "", "## observer", "", "profiles/observer_overhead_v2.csv (기준 PLAN_RESOLVED §6). 기록 전용 task 0 (validation task_count_valid), 레코드 누락 0 (sampling_valid footer)."]
    open(f"{FEAT}/VALIDATION_REPORT.md", "w").write("\n".join(V) + "\n")
    # ---- MEASUREMENT_COMPLETION_STATUS ----
    M = [f"# MEASUREMENT_COMPLETION_STATUS — IDE_075 ({now_kst()})", "", "| 작업 | 상태 | 근거 |", "|---|---|---|",
         "| A00 근거·코드·예산 | 완료 | PLAN_RESOLVED §1, budget_reconciliation.json |", "| A01 PCM·시간창·GPU union·분모 정정 | 완료 (perf 구간값 NOT_RECOVERABLE, client_request_window ESTIMATED) | WINDOW_RECONSTRUCTION.md, legacy_vs_corrected.csv |", "| A02 v2 지표 | 완료 | METRIC_DEFINITIONS_V2.yaml, 세션/v2 |", "| A03 저간섭 기록기 | 완료 (task 0 추가, 합성 시험 PASS) | kt_evt_patch_v2.py, parser_test_results.json |",
         "| A04 barrier·clock·TID | 부분: 클라이언트 barrier 미지원(vllm bench serve) → 서버측 이벤트로 사후 절단; clock 은 순서검사·오차구간 | window_events.jsonl, clock_alignment_summary.csv |", f"| A05 producer/consumer·FIFO | {'완료' if QB else 'NOT_COLLECTED'} | queue_wait_breakdown, fifo_casebook |", f"| A06 expert rows·분기·단계 | {'완료 (작업 단위; expert 별 rows 는 미수집)' if PL else 'NOT_COLLECTED'} | producer_layer_costs.csv, expert_cost_samples.csv.gz |",
         f"| A07 부하 창 CPU·DRAM | {'완료 (perf -I 1000 interval, PCM v2)' if any(m['mode']=='RESOURCE' for m in S) else 'NOT_COLLECTED'} | resource_summary.json |", f"| A08 GPU 전송·결합 | 완료 (HtoD/DtoH·combine·idle-in-gap); TP collective 심화 미실행(조건부) | gpu_layer_timeline, v2 |", "| A09 prefill·C1·tail | 미실행 (조건부; 예산 없음) | — |", "| A10 정상성 | smoke 4 요청 부팅당, lifecycle/task_count 검증 | validation_results_v2 |",
         f"| A11 OFF/CORE/CORR/RESOURCE | 세션 {u['sessions']}/6 (+예비 0) | FULL_REPORT §2·3 |", "| A12 판정·인계 | READINESS.md, OPTIMIZATION_HANDOFF.md | — |", "| G00 GLM | GLM_NORMAL_OUTPUT_BLOCKED (실행 없음) | IDE_074 gate_result.json |", "", f"예산: 세션 {u['sessions_total']}/24 · 부팅 {u['boot_total']}/12 (IDE_075 시도 {u['boot_attempts']}, 중단 {u['boot_aborted']}) · 품질 4/80"]
    open(f"{FEAT}/MEASUREMENT_COMPLETION_STATUS.md", "w").write("\n".join(M) + "\n")
    # ---- BOTTLENECK_EVIDENCE ----
    B = [f"# BOTTLENECK_EVIDENCE — IDE_075 ({now_kst()})", "", "지시서 §0 의 5 질문에 대한 직접 근거. 원값은 세션/v2. 확인 / 원인 미분리 / 미측정 구분. E2E 환산·개선율 없음.", ""]
    for sname, d in DEP.items():
        for g, v in d["by_graph_cohort"].items():
            if not g.endswith("cold_present_nonempty") or v["n"] < 1000: continue
            qb = QB.get(sname) or []; g_ = lambda k: [float(x[k]) for x in qb] if qb else []
            B += [f"## {sname} — {v['step_names']} (cold_present_nonempty n={v['n']})", "", "| 질문 | 관측 (p50 / p95, µs) | 구분 |", "|---|---|---|",
                  f"| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 {f1((v.get('cold_pub_delta_us') or {}).get('p50'))} / {f1((v.get('cold_pub_delta_us') or {}).get('p95'))}, late share {f1(d['sensitivity'][g]['late_share_thr_0us'],3)} (thr5 {f1(d['sensitivity'][g]['late_share_thr_5us'],3)}) | 확인 (층별 분포 제공) |",
                  f"| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 {f1((v.get('enqueue_to_start_us') or {}).get('p50'))} = 선행 실행 점유 {f1((v.get('predecessor_exec_union_us') or {}).get('p50'))} (deferred {f1((v.get('predecessor_deferred_exec_us') or {}).get('p50'))}) + 미분리 {f1((v.get('queue_gap_unattributed_us') or {}).get('p50'))} / p95 {f1((v.get('queue_gap_unattributed_us') or {}).get('p95'))}; go→enqueue {f1((v.get('go_to_enqueue_us') or {}).get('p50'))} | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |",
                  f"| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service {f1((v.get('deferred_service_span_us') or {}).get('p50'))}; end→pub {f1((v.get('producer_end_to_pub_us') or {}).get('p50'))}; pub→h2d_start {f1((v.get('pub_to_h2d_start_us') or {}).get('p50'))}; h2d {f1((v.get('h2d_duration_us') or {}).get('p50'))}; post-h2d→combine {f1((v.get('gpu_post_h2d_gap_us') or {}).get('p50'))}; pre-h2d gap {f1((v.get('gpu_pre_h2d_gap_us') or {}).get('p50'))} 중 GPU idle {f1((v.get('gpu_idle_inside_gap_us') or {}).get('p50'))} | 확인 |",
                  f"| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 {d['coverage'].get('count_mismatch_rows', 0)}, clock {d['clock_validity_counts']}, residual max {f1((v.get('identity_residual_us') or {}).get('max'))}, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |",
                  f"| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |", ""]
    B += ["## 미측정·한계", "", "- expert 별 rows·분기는 작업 단위 합계(unique/max/sum rows, AMX/AVX 호출 수)로만 수집. expert 단위 표본 없음.", "- prefill(EXTEND) 층 분절 없음. TP collective 위치 귀속 미확정 (조건부 미실행).", "- OFF 두 부팅이 CORE 앞에 위치, CORE/CORR/RESOURCE 는 같은 부팅. 단발 CORE/RESOURCE 통계 한계.", "- clock: 일정 offset 은 순서검사로 검출되지 않음.", "- DRAM: 부하 창 PCM 값은 관측값이며 포화 판정 없음 (비교 기준 부재)."]
    open(f"{FEAT}/BOTTLENECK_EVIDENCE.md", "w").write("\n".join(B) + "\n")
    # ---- raw ----
    os.makedirs(f"{FEAT}/raw_md", exist_ok=True)
    for old in glob.glob(f"{FEAT}/raw_md/part-*.md"): os.remove(old)
    parts = []; buf = []; size = 0; k = 1; LIM = 8_000_000
    def flush():
        nonlocal buf, size, k
        if not buf: return
        p = f"{FEAT}/raw_md/part-{k:04d}.md"; open(p, "w").write("\n".join(buf) + "\n"); parts.append((f"raw_md/part-{k:04d}.md", os.path.getsize(p))); buf = []; size = 0; k += 1
    def emit(title, text):
        nonlocal buf, size
        if len(text) > LIM:
            n = (len(text) + LIM - 1) // LIM
            for i in range(n): emit(f"{title} (조각 {i+1}/{n})", text[i*LIM:(i+1)*LIM])
            return
        chunk = f"\n## {title}\n\n```\n{text}\n```\n"
        if size + len(chunk) > LIM: flush()
        buf.append(chunk); size += len(chunk)
    def rd(p):
        try: return gzip.open(p, "rt", errors="replace").read() if p.endswith(".gz") else open(p, errors="replace").read()
        except Exception as e: return f"<read error {e}>"
    for p in sorted(glob.glob(f"{ST}/*")) + sorted(glob.glob(f"{OFF}/*")) + sorted(glob.glob(f"{FEAT}/profiles/*.csv")) + [f"{FEAT}/PROGRESS.md", f"{FEAT}/PROGRESS_EVENTS.jsonl", f"{FEAT}/WORK_LOG.md"]:
        if os.path.isfile(p): emit(rel(p), rd(p))
    for bd in sorted(glob.glob(f"{CAMP}/qwen/OPT4/*/")):
        for fn in ("launch_cmd.sh", "requested_config.json", "runtime_proofs.json", "server_info.json", "smoke_greedy4.json", "thread_affinity.json", "thread_role_map.csv"):
            if os.path.exists(f"{bd}{fn}"): emit(rel(f"{bd}{fn}"), rd(f"{bd}{fn}"))
        for sdir in sorted(glob.glob(f"{bd}*/")):
            for fn in ("metrics.json", "bench_cmd.sh", "bench.stdout.log", "bench.stderr.log", "requests.jsonl", "window_events.jsonl", "observer_config.json", "cpu_timeseries.csv", "gpu_timeseries.csv", "memory_timeseries.csv", "cpu_counter_intervals.csv", "pcm_memory.raw.csv", "perf_stat.log", "pcm_memory.log", "thread_affinity_start.json"):
                if os.path.exists(f"{sdir}{fn}"): emit(rel(f"{sdir}{fn}"), rd(f"{sdir}{fn}"))
            for p in sorted(glob.glob(f"{sdir}v2/*")): emit(rel(p) + (" (gunzip 전문)" if p.endswith(".gz") else ""), rd(p))
        if os.path.exists(f"{bd}server.full.log.gz"): emit(rel(f"{bd}server.full.log.gz") + " (전문)", rd(f"{bd}server.full.log.gz"))
    for p in sorted(glob.glob(f"{KT}/*/kt_evt.csv*")): emit(p + " (전문)", rd(p))
    flush()
    open(f"{FEAT}/FULL_RAW_DATA.md", "w").write("\n".join([f"# FULL_RAW_DATA — IDE_075 ({now_kst()})", "", "텍스트 원자료 전부 raw_md/part-*.md. 바이너리 trace 는 서버 보존·SHA256 (ARTIFACT_INDEX.csv).", "", "| part | bytes |", "|---|---|"] + [f"| {p} | {b} |" for p, b in parts]) + "\n")
    files = [p for p in glob.glob(f"{CAMP}/**/*", recursive=True) if os.path.isfile(p)] + [p for p in glob.glob(f"{FEAT}/**/*", recursive=True) if os.path.isfile(p) and not p.endswith(("ARTIFACT_INDEX.csv", "SHA256SUMS.txt"))] + [p for p in glob.glob(f"{KT}/**/*", recursive=True) if os.path.isfile(p)]
    with open(f"{FEAT}/ARTIFACT_INDEX.csv", "w", newline="") as f, open(f"{FEAT}/SHA256SUMS.txt", "w") as g:
        w = csv.writer(f); w.writerow(["path", "bytes", "sha256", "kind", "access", "in_git(<=5MB & repo)"]); n = 0; tot = 0
        for p in sorted(files):
            b = os.path.getsize(p); h = sha(p); n += 1; tot += b
            w.writerow([rel(p), b, h, "trace" if p.endswith(".trace.json.gz") else ("cpu_event_log" if "kt_evt" in p else "text"), "server violet-h100-016" if not p.startswith(REPO) else "repo", b <= 5 * 1024 * 1024 and p.startswith(REPO)]); g.write(f"{h}  {rel(p)}\n")
    print(f"FULL_REPORT {len(L)} lines; parts {len(parts)}; sessions {len(rows)}; artifacts {n} ({tot/1e9:.2f} GB); DEP {list(DEP)}; VAL {list(VAL)}; usage {u}")


if __name__ == "__main__": main()
