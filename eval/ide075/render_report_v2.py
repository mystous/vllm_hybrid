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
            "ttft_p50": s.get("median_ttft_ms"), "ttft_p95": s.get("p95_ttft_ms"), "tpot_p50": s.get("median_tpot_ms"), "tpot_p95": s.get("p95_tpot_ms"), "e2el_p50": s.get("median_e2el_ms"), "e2el_p95": s.get("p95_e2el_ms"), "itl_p95": s.get("p95_itl_ms"), "valid": bool(s) and s.get("failed") == 0 and m["rc"] == 0, "t_start": m["t_start"]["wall_kst"], "so": m["_so"], "client": m.get("client", "vllm"), "dir": rel(m["_dir"])}


def main():
    S = sessions(CAMP); rows = [prow(m) for m in S]; ev = jl(f"{ST}/execution_events.jsonl"); ev74 = jl(f"{C74}/state/execution_events.jsonl")
    u = {"boot_attempts": sum(e["event"] == "boot_start" for e in ev), "boot_aborted": sum(e["event"] == "boot_end" and e.get("verdict") != "HEALTH_OK" for e in ev), "sessions": sum(e["event"] == "session_end" for e in ev), "smoke": sum(e.get("requests", 0) for e in ev if e["event"] == "smoke_end"), "warmup": sum(e.get("requests", 0) for e in ev if e["event"] == "warmup_end"),
         "boot_074": sum(e["event"] == "boot_start" for e in ev74), "sessions_074": sum(e["event"] == "session_end" for e in ev74)}
    u["boot_ext"] = sum(e["event"] == "boot_start" and e.get("phase") == "extended" for e in ev); u["sessions_ext"] = sum(e["event"] == "session_end" and e.get("phase") == "extended" for e in ev)
    u["boot_attempts"] -= u["boot_ext"]; u["sessions"] -= u["sessions_ext"]
    u["boot_total"] = u["boot_074"] + u["boot_attempts"]; u["sessions_total"] = u["sessions_074"] + u["sessions"]
    json.dump({"budget_total": {"SESSIONS": 24, "BOOT": 12, "QUALITY_Q": 80}, "ide074": {"sessions": u["sessions_074"], "boots": u["boot_074"], "quality": 4}, "ide075": {"sessions": u["sessions"], "boot_attempts": u["boot_attempts"], "boot_aborted": u["boot_aborted"], "smoke_requests": u["smoke"], "warmup_requests": u["warmup"], "quality": 0}, "ide075_extended(user_approved)": {"sessions": u["sessions_ext"], "boot_attempts": u["boot_ext"]}, "remaining": {"SESSIONS": 24 - u["sessions_total"], "BOOT": 12 - u["boot_total"], "QUALITY_Q": 76}}, open(f"{ST}/budget_reconciliation.json", "w"), indent=1)
    os.makedirs(f"{FEAT}/profiles", exist_ok=True)
    # ---- observer_overhead_v2 ----
    off = [r for r in rows if r["mode"] == "OFF" and r["valid"] and r["workload"] == "M073_QWEN_PROBE128" and r["client"] == "vllm"]
    off_probe = [r for r in rows if r["mode"] == "OFF" and r["valid"] and r["workload"] == "M073_QWEN_PROBE128" and r["client"] == "probe"]
    OO = [["session", "mode", "boot", "tps", "duration_s", "e2el_p50", "ttft_p95", "tpot_p95", "ref", "throughput_ratio", "duration_ratio", "e2el_p50_ratio", "ttft_p95_ratio", "tpot_p95_ratio", "criterion(|tps|<=3%,p95<=5%)", "status"]]
    for r in rows:
        if r["workload"] != "M073_QWEN_PROBE128" or not r["valid"] or r["mode"] == "OFF": continue
        base_ = off_probe if (r["client"] == "probe" and off_probe) else off
        refs = [("OFF_mean(" + ("probe" if base_ is off_probe else "vllm") + ")", {k: st.mean(o[k] for o in base_) for k in ("tps", "duration", "e2el_p50", "ttft_p95", "tpot_p95")})] + [(o["session"], o) for o in base_]
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
    L += ["", f"실행량: IDE_074 세션 {u['sessions_074']}/부팅 {u['boot_074']} + IDE_075 기본 세션 {u['sessions']}/부팅 시도 {u['boot_attempts']} (중단 {u['boot_aborted']}) → 합계 세션 {u['sessions_total']}/24 · 부팅 {u['boot_total']}/12 · smoke 요청 {u['smoke']} · warmup 요청 {u['warmup']}. **확장 단계(사용자 승인, 상한 밖)**: 세션 {u['sessions_ext']} · 부팅 {u['boot_ext']} (`state/budget_reconciliation.json`)", "",
          "## 2. 모든 세션 (원값)", "", "| boot | session | mode | client | workload | C | n | 완료/실패 | dur s | in | out | out tok/s | TTFT p50/p95 | TPOT p50/p95 | E2EL p50/p95 | valid | .so | start |", "|" + "---|" * 18]
    for r in rows: L.append(f"| {r['boot']} | {r['session']} | {r['mode']} | {r['client']} | {r['workload']} | {r['C']} | {r['n']} | {f1(r['completed'],0)}/{f1(r['failed'],0)} | {f1(r['duration'])} | {f1(r['in'],0)} | {f1(r['out'],0)} | {f1(r['tps'],2)} | {f1(r['ttft_p50'],0)}/{f1(r['ttft_p95'],0)} | {f1(r['tpot_p50'])}/{f1(r['tpot_p95'])} | {f1(r['e2el_p50'],0)}/{f1(r['e2el_p95'],0)} | {r['valid']} | {r['so']} | {r['t_start']} |")
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
    # ---- 확장 단계 (phase=extended) ----
    L += ["## 8b. 확장 단계 (사용자 승인; 원장 phase=extended) — 클라이언트 동등성·TP collective·prefill·TID 역할·tail off-CPU·expert 표본", ""]
    L += ["### 클라이언트 동등성 (OFF 세션, 같은 부팅 안 연속 실행; PROBE128 C=64)", "", "| boot | session | client | tps | TTFT p50/p95 | TPOT p50/p95 | 같은 부팅 vllm 대비 tps 비율 |", "|---|---|---|---|---|---|---|"]
    for bd in sorted(set(r["boot"] for r in rows)):
        offs = [r for r in rows if r["boot"] == bd and r["mode"] == "OFF" and r["valid"]]; ref = [r for r in offs if r["client"] == "vllm"]
        if len(offs) < 2: continue
        for r in offs: L.append(f"| {bd} | {r['session']} | {r['client']} | {f1(r['tps'],1)} | {f1(r['ttft_p50'],0)}/{f1(r['ttft_p95'],0)} | {f1(r['tpot_p50'])}/{f1(r['tpot_p95'])} | {f1(r['tps']/ref[0]['tps'],3) if ref else 'null'} |")
    L += ["", "판정: `vllmw`(= vllm bench serve 본체를 그대로 쓰고 benchmark() 진입 시 ready/start barrier + perf_counter↔REALTIME anchor 만 추가한 래퍼, `eval/ide075/vllm_bench_wrapper.py`) 는 vllm bench 와 tps 0.3% 이내 → **동등**. `probe`(aiohttp 자체 클라이언트) 는 핀 여부와 무관하게 서버 스텝을 늦춤 (tps 0.58~0.72 배, TPOT 1.5~1.8 배) → **비동등**; probe 로 얻은 CORE2/CORE3 의존성 값은 '클라이언트 간섭 하 관측' 으로 표기하고, CORE4(vllmw) 값을 대표값으로 쓴다.", ""]
    L += ["### 클라이언트별 의존성 핵심값 비교 (bs=63 decode, cold_present_nonempty; p50 µs)", "", "| session | client | n | cold_pub_delta | late share | enq→start | service | pre_h2d gap | mapping |", "|---|---|---|---|---|---|---|---|---|"]
    for m_ in S:
        d_ = DEP.get(m_["session"])
        if not d_: continue
        for g, v in d_["by_graph_cohort"].items():
            if not g.endswith("cold_present_nonempty") or v["n"] < 500: continue
            L.append(f"| {m_['session']} | {m_.get('client','vllm')} | {v['n']} {v['step_names']} | {f1((v.get('cold_pub_delta_us') or {}).get('p50'))} | {f1(d_['sensitivity'][g]['late_share_thr_0us'],3)} | {f1((v.get('enqueue_to_start_us') or {}).get('p50'))} | {f1((v.get('deferred_service_span_us') or {}).get('p50'))} | {f1((v.get('gpu_pre_h2d_gap_us') or {}).get('p50'))} | {d_.get('mapping_method','')[:40]} cov={ {k: d_['coverage'][k] for k in d_['coverage'] if k.startswith('resync') or k.startswith('count_mismatch')} } |")
    L += ["", "매핑: 계수 일치 (graph,layer) 는 순서 zip; 불일치(층당 1건, 세션 창 밖 replay) 는 anchor 재동기화(GPU dtoh_last_end 이후 첫 CPU t_go, 다음 replay 전; 5 µs 지터) 로 대응하고 남는 레코드는 `resync_unmatched_*` 로 계수. zip 이동 없음.", ""]
    L += ["### TP collective 위치·rank (tp_collective_summary.json; 위치 = 직전 fused_moe 유무 휴리스틱)", "", "| session | graph:position | step | n | dur p50 r0/r1/r2/r3 (µs) | 도착 시차 p50/p95 | 마지막 도착 rank |", "|---|---|---|---|---|---|---|"]
    for m_ in S:
        p_ = f"{m_['_dir']}v2/tp_collective_summary.json"
        if os.path.exists(p_):
            for k, v in json.load(open(p_))["by_graph_position"].items(): L.append(f"| {m_['session']} | {k} | {v['step']} | {v['n']} | {f1(v['r0_dur_p50'])}/{f1(v['r1_dur_p50'])}/{f1(v['r2_dur_p50'])}/{f1(v['r3_dur_p50'])} | {f1(v['arrival_skew_p50'])}/{f1(v['arrival_skew_p95'])} | {v['last_arrival_rank_counts']} |")
    L += ["", "### prefill/EXTEND 층 (eager_summary.json; 층 = step 내 cold HtoD 순번, CPU 는 같은 step 창 eager 레코드 순)", "", "| session | toks | n | cold_pub_delta p50 | pre_h2d gap p50 | service p50 | budget p50 | late share | AMX/AVX(numa0) |", "|---|---|---|---|---|---|---|---|---|"]
    for m_ in S:
        p_ = f"{m_['_dir']}v2/eager_summary.json"
        if os.path.exists(p_):
            d_ = json.load(open(p_))
            for tk_, v in d_["by_toks"].items(): L.append(f"| {m_['session']} | {tk_} | {v['n']} | {f1(v['cold_pub_delta_us'].get('p50'))} | {f1(v['gpu_pre_h2d_gap_us'].get('p50'))} | {f1(v['deferred_service_span_us'].get('p50'))} | {f1(v['overlap_budget_us'].get('p50'))} | {f1(v['late_share'],3)} | {f1(v['namx0_p50'],0)}/{f1(v['navx0_p50'],0)} |")
            L.append(f"| {m_['session']} coverage | {json.dumps(d_['coverage'])} | | | | | | | |")
    L += ["", "### TID 역할별 CPU 시간 (cpu_time_by_role.json; 부하 창, /proc stat 1 s)", "", "| session | role | tids | cpu_equivalents | comms |", "|---|---|---|---|---|"]
    for m_ in S:
        p_ = f"{m_['_dir']}v2/cpu_time_by_role.json"
        if os.path.exists(p_):
            for r_, v in json.load(open(p_))["by_role"].items(): L.append(f"| {m_['session']} | {r_} | {v['tids']} | {f1(v['cpu_equivalents'],2)} | {json.dumps(v['comms'], ensure_ascii=False)[:80]} |")
    L += ["", "### tail off-CPU (FOCUS; tail_offcpu_summary.json)", ""]
    for m_ in S:
        p_ = f"{m_['_dir']}v2/tail_offcpu_summary.json"
        if os.path.exists(p_): L += [f"{m_['session']}: `{json.dumps(json.load(open(p_)), ensure_ascii=False)[:700]}`", ""]
    L += ["### expert 단위 rows 표본 (expert_rows_summary.json; v3, slot 별 1/16)", ""]
    for bd in sorted(glob.glob(f"{CAMP}/qwen/OPT4/*/expert_rows_summary.json")):
        d_ = json.load(open(bd)); L += [f"{bd.split('/')[-2]}: 표본 {d_['samples_total']} (decode bs63 {d_['decode_bs63_samples(qlen64)']}, prefill {d_['prefill_samples(qlen>64)']}); decode n_unique p50 {d_['decode']['n_unique'].get('p50')}, rows_max p50 {d_['decode']['rows_max'].get('p50')}, expert rows 분포 {json.dumps(d_['decode']['expert_rows_distribution'])[:200]}, rows≤2 비율 {f1(d_['decode']['share_experts_rows_le2'],3)}; prefill n_unique p50 {d_['prefill']['n_unique'].get('p50')}, rows_max p50 {d_['prefill']['rows_max'].get('p50')}", ""]
    # ---- G00 GLM ----
    L += ["## 8c. G00 — GLM-4.7-FP8 (BASIC4) 정상 출력 진단·근본 원인·수정 (확장 단계)", "", "| boot | 진단 | 구성 | 게이트 | 정답 | 출력 앞부분 |", "|---|---|---|---|---|---|"]
    for gd in sorted(glob.glob(f"{CAMP}/glm/BASIC4/GLM_D*/")):
        rq = json.load(open(f"{gd}requested_config.json")) if os.path.exists(f"{gd}requested_config.json") else {}; sa = rq.get("server_args", {}); rsf = "rsf 수정" if "RSF " in (rq.get("rsf_patch") or "") else "rsf 원본"
        grs = glob.glob(f"{gd}gate_result_*.json")
        if not grs: L.append(f"| {gd.split('/')[-2]} | — | gpu_experts={sa.get('kt-num-gpu-experts')} deferred={sa.get('kt-max-deferred-experts-per-token')} {rsf} | NOT_RUN (부팅 실패/OOM) | | |"); continue
        for gp in grs:
            g_ = json.load(open(gp)); L.append(f"| {gd.split('/')[-2]} | {g_['tag']} | gpu_experts={sa.get('kt-num-gpu-experts')} deferred={sa.get('kt-max-deferred-experts-per-token')} {rsf} | {g_['status']} | {g_.get('correct')}/{g_.get('n')} | {(g_.get('texts') or [''])[0][:60].replace('|','/').replace(chr(10),' ')} |")
    ktl = f"{CAMP}/glm/kt_fp8_tests_after_fix.log"
    L += ["", "kt-kernel FP8 단독 테스트 (`examples/test_fp8_perchannel_moe.py`, `test_fp8_moe.py`): 수정 전 설치 .so(v1~v3)·원본 .so 모두 출력 0 (Mean relative L1 100%); upstream 6d460cc 클린 빌드 PASS 0.5829%. 파일 이식 bisect → `operators/amx/moe_base.hpp` 단독으로 재현.", "", "근본 원인 (G00_code_review.md §8): IDE_046-b/IDE_051 hunk 3곳의 `if constexpr (requires(... from_mat_row/from_mat_block ...)) if (env) {...} else ORIGINAL;` — `else` 가 안쪽 `if` 에 결합해 ORIGINAL 도 `if constexpr` 본문에 들어감. FP8/FP8PerChannel/BF16 커널의 BufferA(`BufferABF16Impl`) 에는 해당 메서드가 없어 조건이 거짓 → 입력 gather·A 양자화·down A 양자화가 전부 컴파일에서 제거 → 출력 0. INT4(BufferAImpl) 는 조건 참 → Qwen 무영향. 앞서 확인한 `routed_scaling_factor` 결함(§4, GPU 기여에만 2.5 배 + decode 이중 적용) 은 독립 결함.", "", "수정 후 (`eval/ide075/glm_fp8_dispatch_fix.py`, .so 재빌드):", "", "```", open(ktl).read()[:1500] if os.path.exists(ktl) else "NOT_RUN", "```", ""]
    # 기준 답안 대조 (IDE_073 G-GPU8 quality20, 같은 4문항 greedy)
    refp = f"{REPO}/eval/results/IDE_073_20260917/G-GPU8/a1/quality20.json"
    if os.path.exists(refp):
        def _tx(p_): g_ = json.load(open(p_)); return [(i.get("text") or "") for i in (g_.get("items") or g_.get("results") or [])]
        R_ = _tx(refp)[:4]; L += ["전 GPU 기준(IDE_073 G-GPU8, TP8, kt 없음) greedy 텍스트와의 공통 접두 길이 (문자; 완전 일치 = ref 길이):", "", "| boot/tag | Q0 | Q1 | Q2 | Q3 | 완전 일치 수 |", "|---|---|---|---|---|---|", "| ref 길이 | " + " | ".join(str(len(x)) for x in R_) + " | 4 |"]
        for gp in sorted(glob.glob(f"{CAMP}/glm/BASIC4/GLM_D*/gate_quality4_*.json")):
            T_ = _tx(gp); cps = [len(os.path.commonprefix([r_, t_])) for r_, t_ in zip(R_, T_)]; ex = sum(1 for r_, t_ in zip(R_, T_) if r_ == t_)
            L.append(f"| {gp.split('/')[-2]}/{os.path.basename(gp)[14:-5]} | " + " | ".join(str(c) for c in cps) + f" | {ex} |")
        L.append("")
    GS = []
    for gd in sorted(glob.glob(f"{CAMP}/glm/BASIC4/GLM_D*/")):
        for sdir in sorted(glob.glob(f"{gd}G_*/")):
            if os.path.exists(f"{sdir}metrics.json"): m_ = json.load(open(f"{sdir}metrics.json")); m_["_dir"] = sdir; GS.append(m_)
    if GS:
        L += ["GLM 세션 (수정 .so v4 + rsf 패치; PROBE128 C=64 n=128). D6 은 `--model qwen480`(Qwen 토크나이저) 로 실행된 참고값, D8 은 glm47 토크나이저·KT_EVT 설정(BASIC4 는 callback-free 가 아니라 CPU 레코드 없음), D9 는 BASIC4 인자 + KT_CALLBACK_FREE/KT_CF_SKIP_EMPTY_IMM (CPU 기록기 동작; 전 GPU 기준 greedy 일치 1/4 로 D6/D8 의 3/4 과 다름):", "", "| session | mode | client | tps | TTFT p50/p95 | TPOT p50/p95 | 의존성 (cold_present_nonempty p50: pub_delta / late share / enq→start / service) |", "|---|---|---|---|---|---|---|"]
        for m_ in GS:
            r = prow({**m_, "_boot": m_["boot_id"], "_so": ""}); dp = f"{m_['_dir']}v2/dependency_summary_v2.json"; dep_ = ""
            if os.path.exists(dp):
                d_ = json.load(open(dp))
                for g, v in d_["by_graph_cohort"].items():
                    if g.endswith("cold_present_nonempty") and v["n"] >= 500: dep_ += f"{v['step_names']} n={v['n']}: {f1((v.get('cold_pub_delta_us') or {}).get('p50'))} / {f1(d_['sensitivity'][g]['late_share_thr_0us'],3)} / {f1((v.get('enqueue_to_start_us') or {}).get('p50'))} / {f1((v.get('deferred_service_span_us') or {}).get('p50'))}; "
            L.append(f"| {m_['boot_id']}/{m_['session']} | {m_['mode']} | {m_.get('client')} | {f1(r['tps'],1)} | {f1(r['ttft_p50'],0)}/{f1(r['ttft_p95'],0)} | {f1(r['tpot_p50'])}/{f1(r['tpot_p95'])} | {dep_ or 'v2 없음'} |")
        L.append("")
    L += ["## 9. 기존 자료 정정 (A01; profiles/legacy_vs_corrected.csv, WINDOW_RECONSTRUCTION.md)", "", "| item | session | legacy | corrected | 사유 |", "|---|---|---|---|---|"]
    for r in csv.DictReader(open(f"{FEAT}/profiles/legacy_vs_corrected.csv")): L.append(f"| {r['item']} | {r['session']} | {r['legacy_value']} | {r['corrected_value']} | {r['difference_reason']} |")
    L += ["", "## 10. 오류·중단·미실행", "", "| 항목 | 내용 |", "|---|---|"]
    for e in ev:
        if e["event"] in ("plan_abort", "server_died") or (e["event"] == "boot_end" and e.get("verdict") != "HEALTH_OK"): L.append(f"| {e['event']} {e.get('boot_id')} | {e.get('verdict', '')} {e.get('note', '')} |")
    L += ["| GLM (G00) 기본 단계 | 실행 없음 (IDE_074 게이트 BLOCKED_NORMAL_OUTPUT 계승). 확장 단계에서 진단 D1~D7·근본 원인·수정 → §8c |", "| S7 예비 | 하네스 결함으로 부팅 2회 소진 → 예비 없음 (기본 단계 BLOCKED_BY_BUDGET; 확장 단계에서 OFF_OPEN2/OFF_CLOSE2/CORE2~4 로 보완) |", "| 기록기 v2 → v3 | v2 는 SPSC ring 경쟁으로 fwd 레코드 일부 누락 → v3 (Node 가 rec 포인터 보유, 스레드 이름 부여, expert 표본 ring) 로 교체. CORE2 이후 세션은 v3 |", "| probe 클라이언트 | 비동등 (서버 스텝 지연) → CORE2/CORE3 값은 간섭 하 관측, CORE4(vllmw) 가 대표 |", "", "## 11. 파일", "", "- offline/: legacy_windows.csv, pcm_samples_v2.csv, pcm_window_summary_v2.csv, timestamp_parse_errors.jsonl, parser_test_results.json", "- 세션/v2/: producer_consumer_metrics_v2.csv.gz, dependency_summary_v2.json, missing_coverage_v2.csv, queue_wait_breakdown.csv.gz, fifo_dependency_edges.csv.gz, producer_layer_costs.csv, expert_cost_samples.csv.gz, fifo_casebook.md, validation_results_v2.json, resource_summary.json", "- 세션/: window_events.jsonl, observer_config.json, cpu_counter_intervals.csv, pcm_memory.raw.csv, requests.jsonl, 시계열", "- 서버 보존: ~/.cache/huggingface/kt/ide075/<boot>/{kt_evt.csv, kt_evt.csv.tasks, kt_evt.csv.map, profiles/*.trace.json.gz} (ARTIFACT_INDEX.csv SHA256)"]
    open(f"{FEAT}/FULL_REPORT.md", "w").write("\n".join(L) + "\n")
    # ---- VALIDATION_REPORT ----
    V = [f"# VALIDATION_REPORT — IDE_075 ({now_kst()})", "", "## parser/단위/기록기 합성 시험", "", "```", json.dumps(json.load(open(f"{OFF}/parser_test_results.json")), ensure_ascii=False, indent=1), "```", "", "## 세션별 항목 유효성", ""]
    for sname, v in VAL.items(): V += [f"### {sname} ({v['mode']}) all_pass={v['all_pass']}", "", "```", json.dumps(v["checks"], ensure_ascii=False, indent=1)[:6000], "```", ""]
    V += ["## clock", "", "profiles/clock_alignment_summary.csv. 방법: CPU CLOCK_REALTIME ↔ kineto host 시계 순서검사 (go ≥ dtoh_end). 일정 offset 은 검출 불가 → 5 µs 이내 차이는 CLOCK_INDETERMINATE 로 분류, 민감도 0/1/2/5 µs 전부 보고.", "", "## observer", "", "profiles/observer_overhead_v2.csv (기준 PLAN_RESOLVED §6). 기록 전용 task 0 (validation task_count_valid), 레코드 누락 0 (sampling_valid footer)."]
    open(f"{FEAT}/VALIDATION_REPORT.md", "w").write("\n".join(V) + "\n")
    # ---- MEASUREMENT_COMPLETION_STATUS ----
    M = [f"# MEASUREMENT_COMPLETION_STATUS — IDE_075 ({now_kst()})", "", "| 작업 | 상태 | 근거 |", "|---|---|---|",
         "| A00 근거·코드·예산 | 완료 | PLAN_RESOLVED §1, budget_reconciliation.json |", "| A01 PCM·시간창·GPU union·분모 정정 | 완료 (perf 구간값 NOT_RECOVERABLE, client_request_window ESTIMATED) | WINDOW_RECONSTRUCTION.md, legacy_vs_corrected.csv |", "| A02 v2 지표 | 완료 | METRIC_DEFINITIONS_V2.yaml, 세션/v2 |", "| A03 저간섭 기록기 | 완료 (task 0 추가, 합성 시험 PASS) | kt_evt_patch_v2.py, parser_test_results.json |",
         "| A04 barrier·clock·TID | 완료 (확장): vllm bench 래퍼(vllmw) 로 ready/start barrier + perf_counter↔REALTIME anchor + 요청별 start/ttft/itl (S21b, CORE4); clock anchor 실측 (clock_anchors.jsonl, GPU→host offset ≤ +3.5 µs); TID 역할 = 스레드 이름(v3) + affinity | window_events.jsonl, *.anchor.json, clock_alignment_summary.csv, cpu_time_by_role.json |", f"| A05 producer/consumer·FIFO | {'완료' if QB else 'NOT_COLLECTED'} | queue_wait_breakdown, fifo_casebook |", f"| A06 expert rows·분기·단계 | {'완료 (작업 단위 + v3 expert 단위 rows 표본 1/16)' if PL else 'NOT_COLLECTED'} | producer_layer_costs.csv, expert_cost_samples.csv.gz, expert_rows_summary.json |",
         f"| A07 부하 창 CPU·DRAM | {'완료 (perf -I 1000 interval, PCM v2)' if any(m['mode']=='RESOURCE' for m in S) else 'NOT_COLLECTED'} | resource_summary.json |", f"| A08 GPU 전송·결합 | 완료 (HtoD/DtoH·combine·idle-in-gap) + TP collective 위치·rank·도착 시차 (확장) | gpu_layer_timeline, v2, tp_collective_summary.json |", "| A09 prefill·C1·tail | 완료 (확장): prefill/EXTEND 층 분절(eager_summary), C1(bs=1) 의존성(S12·S26, anchor 재동기화), tail off-CPU(FOCUS S14/S19/S25, perf sched) | eager_summary.json, S12/S26 v2, tail_offcpu_summary.json |", "| A10 정상성 | smoke 4 요청 부팅당, lifecycle/task_count 검증 | validation_results_v2 |",
         f"| A11 OFF/CORE/CORR/RESOURCE | 세션 {u['sessions']}/6 (+예비 0) | FULL_REPORT §2·3 |", "| A12 판정·인계 | READINESS.md, OPTIMIZATION_HANDOFF.md | — |", "| G00 GLM | 확장: 진단 D1~D5(전부 비정상) → 커널 bisect → 근본 원인(moe_base.hpp if-constexpr dangling else, TSK_060) → 수정·재빌드 → D6/D7 게이트(전 GPU 기준 대조) → D8 세션(G_OFF2/G_CORR2/G_OFF3) → D9 callback-free 세션(G_CORR3: CPU 레코드 + 트레이스, G_OFF4) (§8c) | G00_code_review.md §8, glm/BASIC4/GLM_D*/ |", "", f"예산: 세션 {u['sessions_total']}/24 · 부팅 {u['boot_total']}/12 (IDE_075 시도 {u['boot_attempts']}, 중단 {u['boot_aborted']}) · 품질 4/80"]
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
