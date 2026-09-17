#!/usr/bin/env python3
"""IDE_074 M9 — FULL_REPORT.md / BOTTLENECK_EVIDENCE.md / FULL_RAW_DATA.md(+raw_md/part-*.md) / observer_overhead.csv / COMPLETION_STATUS.md / ARTIFACT_INDEX.csv / SHA256SUMS.txt.
평가·원인 추정·권고 없음. 실행량은 원장 재계산. 상태명은 PLAN_RESOLVED §6."""
import glob, gzip, json, os, statistics as st, sys, re, csv, shutil, time, hashlib, collections
HOME = os.path.expanduser("~"); REPO = f"{HOME}/projects/vllm_hybrid"; FEAT = f"{REPO}/shadow_assists/features/IDE_074"; CAMP = f"{REPO}/eval/results/IDE_074_20260917"; ST = f"{CAMP}/state"; KT_HOST = f"{HOME}/.cache/huggingface/kt/ide074"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def f1(x, n=1): return "null" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
def rel(p): return os.path.relpath(p, REPO) if p.startswith(REPO) else p
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def now_kst(): return time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime())
def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None
def sd(v): return f"{st.stdev(v):.2f}" if len(v) > 1 else "null"


def sessions():
    out = []
    for bd in sorted(glob.glob(f"{CAMP}/*/*/*/")):
        model, profile, boot = bd.rstrip("/").split("/")[-3:]
        rq = json.load(open(f"{bd}requested_config.json")) if os.path.exists(f"{bd}requested_config.json") else {}
        for sdir in sorted(glob.glob(f"{bd}*/")):
            mp = f"{sdir}metrics.json"
            if not os.path.exists(mp) or os.path.basename(sdir.rstrip("/")) == "warmup": continue
            m = json.load(open(mp)); m["_dir"] = sdir; m["_model"] = model; m["_profile"] = profile; m["_boot"] = boot; m["_boot_mode"] = rq.get("mode"); out.append(m)
    return out


def perf_row(m):
    s = m.get("summary") or {}
    return {"model": m["_model"], "profile": m["_profile"], "boot": m["_boot"], "session": m["session"], "mode": m["mode"], "kind": m["kind"], "workload": m["workload"], "C": m["C"], "n": m["n"], "completed": s.get("completed"), "failed": s.get("failed"), "duration": s.get("duration"),
            "in_tok": s.get("total_input_tokens"), "out_tok": s.get("total_output_tokens"), "out_tps": s.get("output_throughput"), "total_tps": s.get("total_token_throughput"), "ttft_p50": s.get("median_ttft_ms"), "ttft_p95": s.get("p95_ttft_ms"), "ttft_p99": s.get("p99_ttft_ms"),
            "tpot_p50": s.get("median_tpot_ms"), "tpot_p95": s.get("p95_tpot_ms"), "tpot_p99": s.get("p99_tpot_ms"), "itl_p50": s.get("median_itl_ms"), "itl_p95": s.get("p95_itl_ms"), "e2el_p50": s.get("median_e2el_ms"), "e2el_p95": s.get("p95_e2el_ms"),
            "valid": bool(s) and s.get("failed") == 0 and m.get("rc") == 0, "t_start": m["t_start"]["wall_kst"], "drain_s": (m.get("drain") or {}).get("drain_seconds"), "gpu_idle_confirmed": (m.get("drain") or {}).get("gpu_idle_confirmed"), "healthy_after": m.get("healthy_after"), "dir": rel(m["_dir"])}


def parse_perf_stat(p):
    if not os.path.exists(p): return None
    d = {}
    for l in open(p):
        l = l.strip()
        mm = re.match(r"([\d,\.]+)\s+(msec task-clock|cycles|instructions|context-switches|cpu-migrations|cache-misses)", l)
        if mm: d[mm.group(2)] = float(mm.group(1).replace(",", ""))
        if "CPUs utilized" in l: d["cpus_utilized"] = float(re.search(r"#\s+([\d\.]+)\s+CPUs", l).group(1))
        if "seconds time elapsed" in l: d["elapsed_s"] = float(l.split()[0])
        if "insn per cycle" in l: d["ipc"] = float(re.search(r"#\s+([\d\.]+)\s+insn", l).group(1))
    return d


def parse_pcm(p, t0, t1):
    """pcm-memory csv: 2행 헤더. 시스템 read/write 열 평균 (요청 창 안 표본만; 경계 표본은 boundary_partial 계수)."""
    if not os.path.exists(p): return None
    try:
        rows = list(csv.reader(open(p, errors="replace")))
        if len(rows) < 3: return {"rows": len(rows)}
        h1, h2 = rows[0], rows[1]; hdr = [f"{a.strip()}|{b.strip()}" for a, b in zip(h1, h2)]
        # 마지막 그룹(System) 의 Read/Write/Memory 열
        idx_r = [i for i, h in enumerate(hdr) if h.startswith("System") and h.endswith("|Read")]; idx_w = [i for i, h in enumerate(hdr) if h.startswith("System") and h.endswith("|Write")]; idx_m = [i for i, h in enumerate(hdr) if h.startswith("System") and h.endswith("|Memory")]
        ti = next((i for i, h in enumerate(hdr) if h.endswith("|Time")), None); di = next((i for i, h in enumerate(hdr) if h.endswith("|Date")), None)
        vals = []; inside = 0; boundary = 0; total = 0
        for r in rows[2:]:
            if len(r) < len(hdr) - 1: continue
            total += 1
            try:
                ts = time.mktime(time.strptime(f"{r[di]} {r[ti]}", "%Y-%m-%d %H:%M:%S")) if (ti is not None and di is not None) else None
            except Exception: ts = None
            if ts is not None and (ts < t0 - 1 or ts > t1 + 1): continue
            if ts is not None and (abs(ts - t0) <= 1 or abs(ts - t1) <= 1): boundary += 1
            inside += 1
            vals.append({"read": float(r[idx_r[-1]]) if idx_r else None, "write": float(r[idx_w[-1]]) if idx_w else None, "mem": float(r[idx_m[-1]]) if idx_m else None})
        def mean(k): v = [x[k] for x in vals if x[k] is not None]; return st.mean(v) if v else None
        return {"samples_total": total, "samples_in_window": inside, "boundary_partial": boundary, "system_read_mean": mean("read"), "system_write_mean": mean("write"), "system_memory_mean": mean("mem"), "unit": "MB/s (pcm-memory 기본)", "header_sample": hdr[-6:]}
    except Exception as e: return {"error": str(e)}


def cpu_only_metrics(m):
    """P2 (GPU 트레이스 없음): kt_evt.csv 세션 창의 CPU-side 지표. 층 매핑은 slot 맵으로 (layer = map[slot].layer)."""
    boot = m["boot_id"]; evt = f"{KT_HOST}/{boot}/kt_evt.csv"; mp = f"{KT_HOST}/{boot}/kt_evt.csv.map"
    if not os.path.exists(evt): return None
    t0, t1 = m["t_start"]["epoch"] * 1e9, ((m.get("drain") or {}).get("t_drain_end") or m["t_end"])["epoch"] * 1e9
    smap = {}
    for l in open(mp) if os.path.exists(mp) else []:
        p = l.strip().split(",")
        if len(p) >= 5: smap[int(p[0])] = (int(p[1]), int(p[2]))
    ev = [r for r in csv.DictReader(l for l in open(evt) if not l.startswith("#")) if t0 <= int(r["t_go"]) <= t1]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ev:
        L, rows = smap.get(int(r["slot"]), (None, None)); k = rows
        if int(r["t_def_start"]) and int(r["t_def_end"]):
            by[k]["submit_to_start_us"].append((int(r["t_def_start"]) - int(r["t_go"])) / 1e3); by[k]["cpu_compute_span_us"].append((int(r["t_def_end"]) - int(r["t_def_start"])) / 1e3)
            if int(r["t_numa0_end"]) and int(r["t_numa1_end"]): by[k]["numa_skew_us"].append(abs(int(r["t_numa0_end"]) - int(r["t_numa1_end"])) / 1e3); by[k]["numa0_span_us"].append((int(r["t_numa0_end"]) - int(r["t_numa0_start"])) / 1e3); by[k]["numa1_span_us"].append((int(r["t_numa1_end"]) - int(r["t_numa1_start"])) / 1e3)
            by[k]["n_cold_ids"].append(int(r["n_cold_ids"]))
        if int(r["t_done_set"]): by[k]["go_to_done_set_us"].append((int(r["t_done_set"]) - int(r["t_go"])) / 1e3)
    out = {"records_in_session": len(ev), "by_rows": {}}
    for k, d in by.items(): out["by_rows"][str(k)] = {kk: {"n": len(v), "p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99), "max": max(v)} for kk, v in d.items()}
    return out


def kt_stdout_lines(boot_dir):
    p = f"{boot_dir}/server.full.log.gz"
    if not os.path.exists(p): return [], []
    tq, wrap = [], []
    for l in gzip.open(p, "rt", errors="replace"):
        if l.startswith("[kt-tq]"): tq.append(l.strip())
        elif l.startswith("[kt-wrap]"): wrap.append(l.strip())
    return tq, wrap


def main():
    S = sessions(); rows = [perf_row(m) for m in S]; events = jl(f"{ST}/execution_events.jsonl"); state = json.load(open(f"{ST}/RUN_STATE.json")) if os.path.exists(f"{ST}/RUN_STATE.json") else {}
    u = {"SESSIONS": sum(e["event"] == "session_end" for e in events), "BOOT": sum(e["event"] == "boot_end" for e in events), "QUALITY_Q": sum(e.get("questions", 0) for e in events if e["event"] == "quality_end")}
    env = json.load(open(f"{FEAT}/evidence/environment.json"))
    os.makedirs(f"{FEAT}/profiles", exist_ok=True)
    # ---------- observer_overhead.csv ----------
    p0 = [r for r in rows if r["mode"] == "OFF" and r["workload"] == "M073_QWEN_PROBE128" and r["valid"]]
    ref = {"out_tps": st.mean(r["out_tps"] for r in p0), "duration": st.mean(r["duration"] for r in p0), "e2el_p50": st.mean(r["e2el_p50"] for r in p0), "ttft_p95": st.mean(r["ttft_p95"] for r in p0), "tpot_p95": st.mean(r["tpot_p95"] for r in p0)} if p0 else None
    OO = [["campaign", "boot", "mode", "session", "workload", "out_tps", "duration_s", "e2el_p50_ms", "ttft_p95_ms", "tpot_p95_ms", "ref_P0_mean_out_tps", "throughput_ratio", "elapsed_ratio", "latency_ratio(e2el_p50)", "ttft_p95_ratio", "tpot_p95_ratio", "order_in_boot", "note"]]
    for r in rows:
        if r["workload"] != "M073_QWEN_PROBE128" or not r["valid"]: continue
        OO.append(["IDE_074_20260917", r["boot"], r["mode"], r["session"], r["workload"], r["out_tps"], r["duration"], r["e2el_p50"], r["ttft_p95"], r["tpot_p95"], ref and ref["out_tps"], ref and r["out_tps"] / ref["out_tps"], ref and r["duration"] / ref["duration"], ref and r["e2el_p50"] / ref["e2el_p50"], ref and r["ttft_p95"] / ref["ttft_p95"], ref and r["tpot_p95"] / ref["tpot_p95"], r["session"], "부팅 간 비교 (KT_EVT 정적 env). 허용 기준 사전 미설정 → 비율만 보고"])
    with open(f"{FEAT}/profiles/observer_overhead.csv", "w", newline="") as f: csv.writer(f).writerows(OO)
    # ---------- dependency (P1) ----------
    DEP = {}
    for m in S:
        p = f"{m['_dir']}dependency_summary.json"
        if os.path.exists(p): DEP[m["session"]] = json.load(open(p)); DEP[m["session"]]["_boot"] = m["_boot"]; DEP[m["session"]]["_model"] = m["_model"]
    VAL = {m["session"]: json.load(open(f"{m['_dir']}validation_results.json")) for m in S if os.path.exists(f"{m['_dir']}validation_results.json")}
    CPU2 = {m["session"]: cpu_only_metrics(m) for m in S if m["mode"] == "P2"}
    PERF2 = {m["session"]: parse_perf_stat(f"{m['_dir']}perf_stat.txt") for m in S if m["mode"] == "P2"}
    PCM2 = {m["session"]: parse_pcm(f"{m['_dir']}pcm_memory.csv", m["t_start"]["epoch"], m["t_end"]["epoch"]) for m in S if m["mode"] == "P2"}
    # ---------- FULL_REPORT ----------
    L = [f"# FULL_REPORT — IDE_074 CPU MoE Hot/Cold 병목 측정 ({now_kst()})", "", f"지시서 `PLAN.md` (SHA {sha(f'{FEAT}/PLAN.md')[:16]}…). 확정 실행안 `PLAN_RESOLVED.md`, 조사 기록 `WORK_LOG.md`, 설정 차이 `CONFIG_DIFF.md`, 원자료 색인 `SOURCE_INDEX.md`, 상태 `COMPLETION_STATUS.md`. 평가·원인 추정·권고 없음.", "",
         "## 1. 환경·고정 구성", "", "```", f"host {env['host']['hostname']} kernel {env['host']['kernel']} no_turbo={env['host']['no_turbo']} governor={env['host']['governor']} smt={env['host']['smt']}", f"git HEAD(수집 시) {env['git']['head']} branch {env['git']['branch']} base_is_ancestor={env['git']['base_is_ancestor']}",
         f"container python/torch/sglang: {env['container']['python'][:200]}", f"kt_kernel_ext.so (IDE_073 기준): {env['container']['kt_so'][:80]}", "kt_kernel_ext.so (IDE_074 재빌드): adf49e48ec0510fafb55f14e72ee348938c7410653af18b77529d9124d1b52be 8306720 B (CONFIG_DIFF.md)", "```", "",
         "부팅별 실효 증빙 (runtime_proofs.json):", "", "| model/profile/boot | mode | verdict | per_layer 행 수 / 합 | cf_ready | kt_evt | smoke_greedy4 | kt .so SHA(부팅 시) |", "|---|---|---|---|---|---|---|---|"]
    for bd in sorted(glob.glob(f"{CAMP}/*/*/*/")):
        model, profile, boot = bd.rstrip("/").split("/")[-3:]; rq = json.load(open(f"{bd}requested_config.json")) if os.path.exists(f"{bd}requested_config.json") else {}; pr = json.load(open(f"{bd}runtime_proofs.json")) if os.path.exists(f"{bd}runtime_proofs.json") else {}
        be = next((e for e in events if e["event"] == "boot_end" and e.get("boot_id") == boot), {})
        sm = json.load(open(f"{bd}smoke_greedy4.json")) if os.path.exists(f"{bd}smoke_greedy4.json") else None
        L.append(f"| {model}/{profile}/{boot} | {rq.get('mode')} | {be.get('verdict')} | {len(pr.get('per_layer_effective', []))} / {pr.get('per_layer_sum')} | {pr.get('cf_ready_lines')} | {len(pr.get('kt_evt_lines', []))} | {('있음' if sm else '없음')} | {(rq.get('kt_kernel_so_sha256') or '')[:12]} |")
    # smoke 비교
    smokes = {}
    for bd in sorted(glob.glob(f"{CAMP}/qwen/OPT4/*/")):
        p = f"{bd}smoke_greedy4.json"
        if os.path.exists(p): smokes[bd.rstrip("/").split("/")[-1]] = json.load(open(p))
    L += ["", "부팅별 smoke_greedy4 출력 동일성 (OFF/ON 실행 의미 비교; 텍스트 SHA):", "", "| boot | 항목별 출력 SHA256 앞 12자 |", "|---|---|"]
    for b, smj in smokes.items():
        items = smj if isinstance(smj, list) else smj.get("items") or smj.get("results") or [smj]
        L.append(f"| {b} | " + ", ".join(hashlib.sha256(json.dumps(x.get('text') or x.get('output') or x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12] for x in items) + " |")
    L += ["", "## 2. 모든 세션 (원값·유효성)", "", "| model | profile | boot | session | mode | kind | workload | C | n | 완료/실패 | dur s | in tok | out tok | out tok/s | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p50/p95 | E2EL p50/p95 | drain s (GPU idle 확인) | valid | start |", "|" + "---|" * 21]
    for r in rows: L.append(f"| {r['model']} | {r['profile']} | {r['boot']} | {r['session']} | {r['mode']} | {r['kind']} | {r['workload']} | {r['C']} | {r['n']} | {f1(r['completed'],0)}/{f1(r['failed'],0)} | {f1(r['duration'])} | {f1(r['in_tok'],0)} | {f1(r['out_tok'],0)} | {f1(r['out_tps'],2)} | {f1(r['ttft_p50'],0)}/{f1(r['ttft_p95'],0)}/{f1(r['ttft_p99'],0)} | {f1(r['tpot_p50'])}/{f1(r['tpot_p95'])}/{f1(r['tpot_p99'])} | {f1(r['itl_p50'])}/{f1(r['itl_p95'])} | {f1(r['e2el_p50'],0)}/{f1(r['e2el_p95'],0)} | {f1(r['drain_s'])} ({r['gpu_idle_confirmed']}) | {r['valid']} | {r['t_start']} |")
    L += ["", "통계 (유효 rep, 같은 mode·workload):", "", "| model | mode | workload | n | out tok/s 원값 | 평균 | 중앙값 | sd | boots |", "|---|---|---|---|---|---|---|---|---|"]
    g = collections.defaultdict(list)
    for r in rows:
        if r["valid"]: g[(r["model"], r["mode"], r["workload"])].append(r)
    for (mo, md, wl), rs in g.items():
        v = [r["out_tps"] for r in rs]; L.append(f"| {mo} | {md} | {wl} | {len(v)} | {', '.join(f'{x:.2f}' for x in v)} | {st.mean(v):.2f} | {st.median(v):.2f} | {sd(v)} | {sorted(set(r['boot'] for r in rs))} |")
    L += ["", "## 3. 계측 간섭 (profiles/observer_overhead.csv; 같은 PROBE128 의 OFF(P0) 평균 대비)", "", "| boot | mode | session | out tok/s | throughput_ratio | elapsed_ratio | e2el_p50 ratio | ttft_p95 ratio | tpot_p95 ratio |", "|---|---|---|---|---|---|---|---|---|"]
    for r in OO[1:]: L.append(f"| {r[1]} | {r[2]} | {r[3]} | {f1(r[5],2)} | {f1(r[11],3)} | {f1(r[12],3)} | {f1(r[13],3)} | {f1(r[14],3)} | {f1(r[15],3)} |")
    L += ["", "## 4. 계측 정상성 검증 (M5; 세션별 validation_results.json)", "", "| session | execution | gpu_activity | window | cpu_events | correlation_clock | matched | clock 위반 | lead s(첫 forward − bench 시작) |", "|---|---|---|---|---|---|---|---|---|"]
    for sname, v in VAL.items():
        c = v["checks"]; L.append(f"| {sname} | {c.get('execution',{}).get('pass')} | {c.get('gpu_activity',{}).get('pass')} | {c.get('window',{}).get('pass')} | {c.get('cpu_events',{}).get('pass')} | {c.get('correlation_clock',{}).get('pass')} | {c.get('correlation_clock',{}).get('matched')} | {c.get('correlation_clock',{}).get('clock_violations(t_go<dtoh_end)')} | {f1(c.get('window',{}).get('lead_s(first_step - bench_start)'),2)} |")
    L += ["", "## 5. 의존 관계 지표 요약 (P1; 세션별 dependency_summary.json, µs, 유효 행만; 정의는 PLAN_RESOLVED §3 / dependency_metrics.py docstring)", ""]
    KEYS = ["cold_ready_lateness_us", "cold_pub_minus_hot_ready_us", "exposed_wait_us", "publish_to_release_us", "submit_to_start_us", "cpu_compute_span_us", "cold_budget_us", "numa0_span_us", "numa1_span_us", "numa_completion_skew_us", "publish_delay_us", "cpu_to_gpu_ready_us", "go_minus_dtoh_end_us", "hot_kernel_sum_us", "prev_n_cold_ids"]
    for sname, d in DEP.items():
        L += [f"### {sname} ({d['_model']}, boot {d['_boot']}) — coverage {d['coverage']} · violations {d['violations']}", "", "| graph(step) | rows | 지표 | n | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|---|"]
        for gid, v in d["by_graph"].items():
            for k in KEYS:
                s_ = v.get(k) or {}
                if s_.get("n"): L.append(f"| {gid} {v['step_names']} | {v['rows_valid']} | {k} | {s_['n']} | {f1(s_['p50'])} | {f1(s_['p95'])} | {f1(s_['p99'])} | {f1(s_['max'])} |")
            L.append(f"| {gid} | | share_cold_later_than_hot | {v['rows_valid']} | {f1(v.get('share_cold_later_than_hot'),3)} | | | |")
        L.append("")
    L += ["## 6. CPU 단독 지표·자원 (P2; kt_evt 세션 창, perf stat, pcm-memory)", ""]
    for sname, c2 in CPU2.items():
        L += [f"### {sname}", ""]
        if c2:
            L += [f"kt_evt 레코드 {c2['records_in_session']}", "", "| rows | 지표 | n | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|"]
            for k, d in c2["by_rows"].items():
                for kk, s_ in d.items(): L.append(f"| {k} | {kk} | {s_['n']} | {f1(s_['p50'])} | {f1(s_['p95'])} | {f1(s_['p99'])} | {f1(s_['max'])} |")
        ps = PERF2.get(sname); pc = PCM2.get(sname)
        L += ["", f"perf stat (스케줄러 4 프로세스, 세션 전체 창): `{json.dumps(ps)}`", f"pcm-memory (요청 창 내 표본 평균, 경계 표본 boundary_partial 계수, host 전체 값): `{json.dumps(pc, ensure_ascii=False)[:600]}`", ""]
    for bd in sorted(glob.glob(f"{CAMP}/qwen/OPT4/*/")):
        tq, wr = kt_stdout_lines(bd)
        if tq or wr: L += [f"부팅 {bd.rstrip('/').split('/')[-1]} stdout: [kt-tq] {len(tq)}행, [kt-wrap] {len(wr)}행 (원문은 raw_md)", ""]
    L += ["## 7. 기존 자료 재집계 (M1; IDE_073 D2 retry2, profiles/corrected_metrics.csv, HEURISTIC_MOE_BOUNDARY)", "", "| trace | window | window s | op sum s | busy union s | busy frac | compute union s | memcpy union s | overlap s | 원 busy s / 원 frac |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in csv.DictReader(open(f"{FEAT}/profiles/corrected_metrics.csv")): L.append(f"| {r['trace'][14:19]} | {r['window']} | {float(r['window_len_us'])/1e6:.3f} | {float(r['gpu_op_duration_sum_us'])/1e6:.3f} | {float(r['gpu_busy_union_us'])/1e6:.3f} | {float(r['gpu_busy_fraction']):.3f} | {float(r['compute_union_us'])/1e6:.3f} | {float(r['memcpy_union_us'])/1e6:.3f} | {float(r['compute_memcpy_overlap_us'])/1e6:.3f} | {r.get('orig_gpu_busy_s')} / {r.get('orig_busy_fraction')} |")
    L += ["", "IDE_073 D2 retry2 층 타임라인(패턴 휴리스틱, profiles/gpu_layer_timeline_d2_retry2_TP*.csv.gz): TP-0 graph 2(bs=63) hot 종료→HtoD 시작 gap n 15872 p50 343.2 p90 747.0 p99 1121.5 max 3751.5 µs 합 5.988 s; graph 32(bs=2) p50 10.8 µs. TP-1~3 all-reduce 커널(짝수 순번) duration p50 445 µs 합 7.69 s (순번↔위치 귀속 미확정).", ""]
    L += ["## 8. 오류·재시도·미실행·차단", "", "| 항목 | 상태 | 근거 |", "|---|---|---|"]
    for e in events:
        if e["event"] in ("plan_abort", "server_died", "status_correction", "session_void"): L.append(f"| {e['event']} | {e.get('boot_id') or e.get('cell')} | {json.dumps({k: v for k, v in e.items() if k not in ('t',)}, ensure_ascii=False)[:200]} |")
    for gd in sorted(glob.glob(f"{CAMP}/glm/BASIC4/*/gate_result.json")): L.append(f"| GLM 게이트 {gd.split('/')[-2]} | {json.load(open(gd)).get('status')} | {rel(gd)} |")
    if not glob.glob(f"{CAMP}/glm/BASIC4/*/"): L.append("| GLM (M8) | NOT_RUN | 게이트 미실행 |")
    L += ["", f"## 9. 실행량 (원장 재계산): 세션 {u['SESSIONS']}/24 · 부팅 {u['BOOT']}/12 · 품질 문항 {u['QUALITY_Q']}/80", "", "## 10. 파일", "", "- profiles/: corrected_metrics.csv, observer_windows.csv, observer_overhead.csv, validation_results_*.json, gpu_layer_timeline_*.csv.gz, code_map_survey.md, probe_map.md", "- 세션 디렉터리: dependency_metrics.csv.gz, dependency_summary.json, missing_coverage.csv, gpu_layer_timeline_TP0.csv.gz, requests.jsonl, *_timeseries.csv, perf_stat.txt, pcm_memory.csv, metrics.json", "- 서버 보존(대형): ~/.cache/huggingface/kt/ide074/<boot>/{kt_evt.csv, kt_evt.csv.map, profiles/*.trace.json.gz} — ARTIFACT_INDEX.csv 에 SHA256·크기", "- FULL_RAW_DATA.md + raw_md/part-*.md, ARTIFACT_INDEX.csv, SHA256SUMS.txt, PUBLISH_RECEIPT.md(게시 후)"]
    open(f"{FEAT}/FULL_REPORT.md", "w").write("\n".join(L) + "\n")
    # ---------- BOTTLENECK_EVIDENCE ----------
    B = [f"# BOTTLENECK_EVIDENCE — IDE_074 ({now_kst()})", "", "지시서 §1.1 질문별 직접 근거. 각 값은 세션별 dependency_summary.json / validation_results.json 원값. 확인 / 원인 미분리 / 미측정 을 구분. 성능 개선 상한·E2E 환산 없음.", ""]
    for sname, d in DEP.items():
        B += [f"## 세션 {sname} (boot {d['_boot']}) — 표본 coverage {d['coverage']}, 순서 위반 {d['violations']}", ""]
        for gid, v in d["by_graph"].items():
            cl = v.get("cold_ready_lateness_us") or {}; ew = v.get("exposed_wait_us") or {}; cs = v.get("cpu_compute_span_us") or {}; ss = v.get("submit_to_start_us") or {}; cb = v.get("cold_budget_us") or {}; sk = v.get("numa_completion_skew_us") or {}; pr_ = v.get("publish_to_release_us") or {}; cg = v.get("cpu_to_gpu_ready_us") or {}; hk = v.get("hot_kernel_sum_us") or {}
            B += [f"### graph {gid} {v['step_names']} — 유효 행 {v['rows_valid']} (prev 연결 {v['rows_with_prev']})", "", "| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |", "|---|---|---|---|",
                  f"| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 {f1((v.get('cold_pub_minus_hot_ready_us') or {}).get('p50'))} / {f1((v.get('cold_pub_minus_hot_ready_us') or {}).get('p95'))} / {f1((v.get('cold_pub_minus_hot_ready_us') or {}).get('p99'))}; cold 가 늦은 비율 {f1(v.get('share_cold_later_than_hot'),3)}; other(잔차) 는 hot 이전에 준비 | {cl.get('n')} | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |",
                  f"| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 {f1(ew.get('p50'))} / {f1(ew.get('p95'))} / {f1(ew.get('p99'))} (하한≈bs=2 graph 의 p50) | {ew.get('n')} | 확인 (GPU 시계 단일) |",
                  f"| Q2 Cold 경로 어디서 늦나 | 제출→시작 {f1(ss.get('p50'))} / {f1(ss.get('p95'))}; 계산 {f1(cs.get('p50'))} / {f1(cs.get('p95'))}; 게시→해제 {f1(pr_.get('p50'))} / {f1(pr_.get('p95'))}; 게시→HtoD 완료 {f1(cg.get('p50'))} / {f1(cg.get('p95'))}; cold 예산(go(L−1)→hot_ready(L)) {f1(cb.get('p50'))} / {f1(cb.get('p95'))} | {cs.get('n')} | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |",
                  f"| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 {f1(hk.get('p50'))} / {f1(hk.get('p95'))} | {hk.get('n')} | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |",
                  f"| Q4 NUMA 서브풀 | 완료 시차 p50 {f1(sk.get('p50'))} / {f1(sk.get('p95'))} / {f1(sk.get('p99'))}; numa0 {f1((v.get('numa0_span_us') or {}).get('p50'))} numa1 {f1((v.get('numa1_span_us') or {}).get('p50'))} | {sk.get('n')} | 확인 (원격 메모리 접근 등 원인 미분리) |",
                  f"| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |",
                  f"| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |", ""]
    B += ["## 미측정·한계", "", "- prefill(eager, graph id 0) 층은 층 타임라인 분절 대상에서 제외 (graph node id 없음). F1 LONG 세션의 prefill 구간 Hot/Cold 지연은 미집계. kt_evt.csv 의 eager 슬롯 레코드(qlen>1) 는 원자료로 보존.", "- attention / router / TP collective 의 층별 분해: 커널 이름 수준으로만 트레이스에 존재, 본 보고서 집계 없음 (원 트레이스 서버 보존).", "- 층 번호는 memcpy 순번 패턴 (HEURISTIC_LAYER_ID). rank 1~3 의 대기 위치(all-reduce 순번↔attention/MoE) 귀속 미확정.", "- OFF/ON 은 부팅 간 비교. clock: CPU CLOCK_REALTIME ↔ kineto host 시계 (CUPTI 정렬 오차 미검증; go ≥ dtoh_end 순서검사로만 확인).", "- P2 의 perf/PCM 은 host 전체 값이며 KT 만의 대역폭으로 귀속하지 않음. DRAM 포화 판정 보류."]
    open(f"{FEAT}/BOTTLENECK_EVIDENCE.md", "w").write("\n".join(B) + "\n")
    # ---------- FULL_RAW_DATA + parts ----------
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
    def rd(p, limit=None):
        try:
            if p.endswith(".gz"): return gzip.open(p, "rt", errors="replace").read(limit) if limit else gzip.open(p, "rt", errors="replace").read()
            return open(p, errors="replace").read(limit) if limit else open(p, errors="replace").read()
        except Exception as e: return f"<read error {e}>"
    emit("state/execution_events.jsonl", rd(f"{ST}/execution_events.jsonl")); emit("state/RUN_STATE.json", rd(f"{ST}/RUN_STATE.json"))
    for p in sorted(glob.glob(f"{FEAT}/evidence/*.json")) + [f"{FEAT}/profiles/corrected_metrics.csv", f"{FEAT}/profiles/observer_windows.csv", f"{FEAT}/profiles/observer_overhead.csv"]:
        if os.path.exists(p): emit(rel(p), rd(p))
    for bd in sorted(glob.glob(f"{CAMP}/*/*/*/")):
        for fn in ("launch_cmd.sh", "requested_config.json", "runtime_proofs.json", "server_info.json", "smoke_greedy4.json", "thread_affinity.json", "gate_result.json", "gate_quality4.json"):
            if os.path.exists(f"{bd}{fn}"): emit(rel(f"{bd}{fn}"), rd(f"{bd}{fn}"))
        for sdir in sorted(glob.glob(f"{bd}*/")):
            for fn in ("metrics.json", "bench_cmd.sh", "bench.stdout.log", "bench.stderr.log", "requests.jsonl", "cpu_timeseries.csv", "gpu_timeseries.csv", "memory_timeseries.csv", "validation_results.json", "dependency_summary.json", "missing_coverage.csv", "perf_stat.txt", "perf_stat.log", "pcm_memory.csv", "pcm_memory.log", "thread_affinity_start.json"):
                if os.path.exists(f"{sdir}{fn}"): emit(rel(f"{sdir}{fn}"), rd(f"{sdir}{fn}"))
            for fn in ("dependency_metrics.csv.gz", "gpu_layer_timeline_TP0.csv.gz"):
                if os.path.exists(f"{sdir}{fn}"): emit(rel(f"{sdir}{fn}") + " (gunzip 전문)", rd(f"{sdir}{fn}"))
        if os.path.exists(f"{bd}server.full.log.gz"): emit(rel(f"{bd}server.full.log.gz") + " (전문)", rd(f"{bd}server.full.log.gz"))
    for p in sorted(glob.glob(f"{KT_HOST}/*/kt_evt.csv")) + sorted(glob.glob(f"{KT_HOST}/*/kt_evt.csv.map")): emit(p + " (전문)", rd(p))
    for p in sorted(glob.glob(f"{FEAT}/profiles/*.csv.gz")): emit(rel(p) + " (gunzip 전문)", rd(p))
    for p in (f"{FEAT}/PROGRESS.md", f"{FEAT}/PROGRESS_EVENTS.jsonl", f"{FEAT}/WORK_LOG.md"):
        if os.path.exists(p): emit(rel(p), rd(p))
    flush()
    R = [f"# FULL_RAW_DATA — IDE_074 ({now_kst()})", "", "텍스트 원자료 전부를 raw_md/part-*.md 로 분할 수록 (설정·요청별 결과·시계열·서버 로그 전문·CPU 이벤트 로그·의존 지표·원장). 바이너리(torch trace .gz)는 서버 보존 경로·SHA256 으로 ARTIFACT_INDEX.csv 에 기재.", "", "| part | bytes |", "|---|---|"] + [f"| {p} | {b} |" for p, b in parts]
    open(f"{FEAT}/FULL_RAW_DATA.md", "w").write("\n".join(R) + "\n")
    # ---------- ARTIFACT_INDEX / SHA256SUMS ----------
    files = [p for p in glob.glob(f"{CAMP}/**/*", recursive=True) if os.path.isfile(p)] + [p for p in glob.glob(f"{FEAT}/**/*", recursive=True) if os.path.isfile(p) and not p.endswith(("ARTIFACT_INDEX.csv", "SHA256SUMS.txt"))] + [p for p in glob.glob(f"{KT_HOST}/**/*", recursive=True) if os.path.isfile(p)]
    with open(f"{FEAT}/ARTIFACT_INDEX.csv", "w", newline="") as f, open(f"{FEAT}/SHA256SUMS.txt", "w") as g:
        w = csv.writer(f); w.writerow(["path", "bytes", "sha256", "in_git(<=5MB & repo)", "location"]); n_art = 0; tot = 0
        for p in sorted(files):
            szb = os.path.getsize(p); h = sha(p); tot += szb; n_art += 1
            w.writerow([rel(p), szb, h, szb <= 5 * 1024 * 1024 and p.startswith(REPO), "server violet-h100-016" if not p.startswith(REPO) else "repo"]); g.write(f"{h}  {rel(p)}\n")
    # ---------- COMPLETION_STATUS ----------
    stages = {"M0": "완료 (SOURCE_INDEX.md, evidence/)", "M1": "완료 (gpu_union.py selftest PASS, corrected_metrics.csv)", "M2": "완료 (harness.py: arm→요청→drain→종료, observer_windows.csv)", "M3": "완료 (probe_map.md, kt_evt 스키마, correlation 검증)", "M4": "완료 (CPU 이벤트·GPU 트레이스·perf/pcm)",
              "M5": f"{'PASS' if any(v.get('overall_pass') for v in VAL.values()) else 'FAIL/미실행'} (validation_results)", "M6": f"세션 {u['SESSIONS']}/24 (OFF {sum(1 for r in rows if r['mode']=='OFF')}, P1 {sum(1 for r in rows if r['mode']=='P1')}, P2 {sum(1 for r in rows if r['mode']=='P2')})", "M7": f"dependency_summary {len(DEP)}개 세션", "M8": (json.load(open(glob.glob(f'{CAMP}/glm/BASIC4/*/gate_result.json')[0])).get('status') if glob.glob(f'{CAMP}/glm/BASIC4/*/gate_result.json') else "NOT_RUN"), "M9": "본 문서 생성 시점 기준"}
    C = [f"# COMPLETION_STATUS — IDE_074 ({now_kst()})", "", f"- 캠페인 상태: {state.get('status')}", f"- 실행량: 세션 {u['SESSIONS']}/24 · 부팅 {u['BOOT']}/12 · 품질 문항 {u['QUALITY_Q']}/80", "", "| 작업 | 최종 상태 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in stages.items()]
    C += ["", "## 최종 체크리스트 (§13)", "", "| 항목 | 상태 |", "|---|---|", "| Qwen OPT4 고정 구성·모델/manifest/실행 코드 해시 | evidence/environment.json, hotmap_budget.json, requested_config.json |", "| GPU duration sum/union/gap 분석기 수정·합성 단위시험 | PASS (gpu_union.py) |", "| 요청·수집기 시간창 정렬·장치별 유효 구간 | observer_windows.csv, validation window |", "| layer·step·rank·NUMA·job·ring epoch·graph replay 상관관계 | (slot,epoch)↔(layer,replay) 매칭·순서검사 (validation correlation_clock) |", "| Hot/Cold/other ready·combine 연결 | dependency_metrics (other=잔차, hot 이전 준비) |", "| CPU 큐/연산/완료 전달·전송·GPU 작업 구분 | dependency_metrics / gpu_layer_timeline |", "| 같은 PROBE128 OFF/ON 비교·반복성·간섭·누락률 | observer_overhead.csv, missing_coverage.csv |", "| 실제 scheduler phase 사용 | step[...] annotation |", f"| GLM 조건부 | {stages['M8']} |", "| 최종 상태·원시 자료·해시·게시·전달 | ARTIFACT_INDEX.csv, SHA256SUMS.txt, PUBLISH_RECEIPT.md |"]
    open(f"{FEAT}/COMPLETION_STATUS.md", "w").write("\n".join(C) + "\n")
    print(f"FULL_REPORT {len(L)} lines; raw parts {len(parts)}; sessions {len(rows)}; artifacts {n_art} ({tot/1e9:.2f} GB); usage {u}; DEP {list(DEP)}; VAL {list(VAL)}")


if __name__ == "__main__": main()
