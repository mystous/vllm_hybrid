#!/usr/bin/env python3
"""IDE_072 — 측정 사실만 담은 FULL_REPORT.md / RESULT.md / quality/paired_question_results.md / failures/ / manifests (artifact_manifest, SHA256SUMS, cells.jsonl) 생성.
원장 execution_events.jsonl 에서 실행량 재계산 + 정합성 검사 (§6.1). 평가·해석 없음."""
import glob, gzip, json, os, statistics as st, sys, re
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_072")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
CAMP = f"{REPO}/eval/results/IDE_072_20260916"; OUTF = f"{FEAT}/FULL_REPORT.md"; OUTR = f"{FEAT}/RESULT.md"
for d in ("quality", "failures", "manifests", "reports/request_tables"): os.makedirs(f"{FEAT}/{d}", exist_ok=True)
def f1(x, n=1): return "null" if x is None else f"{x:.{n}f}"
def sd(v): return f"{st.stdev(v):.2f}" if len(v) > 1 else "null"
def rel(p): return os.path.relpath(p, REPO)

def cells():
    out = []
    for cdir in sorted(glob.glob(f"{CAMP}/*/")):
        cid = os.path.basename(cdir.rstrip("/"))
        if cid in ("state", "specs"): continue
        for adir in sorted(glob.glob(f"{cdir}a*/")):
            if os.path.exists(f"{adir}status.json"): out.append((cid, os.path.basename(adir.rstrip("/")), adir, json.load(open(f"{adir}status.json")), json.load(open(f"{adir}cell_spec.json")) if os.path.exists(f"{adir}cell_spec.json") else {}))
    return out

def pooled(rd):
    rq = f"{rd}requests.jsonl"
    if not os.path.exists(rq): return {}
    tt = []; tp = []; n = 0; outs = 0; ins = 0
    for l in open(rq):
        j = json.loads(l); n += 1; outs += j.get("output_len") or 0; ins += j.get("input_len") or 0
        if j.get("ttft_ms") is not None: tt.append(j["ttft_ms"])
        if j.get("tpot_ms") is not None: tp.append(j["tpot_ms"])
    def p(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None
    return {"n": n, "in_sum": ins, "out_sum": outs, "ttft_p95_pooled": p(tt, .95), "tpot_p95_pooled": p(tp, .95), "ttft_p99_pooled": p(tt, .99), "tpot_p99_pooled": p(tp, .99), "algo": "nearest-rank on requests.jsonl"}

def rep_rows(C):
    rows = []
    for cid, att, adir, stt, sp in C:
        for r in stt.get("reps") or []:
            rd = f"{adir}{r['rep_id']}/"; ts = json.load(open(f"{rd}timestamps.json")) if os.path.exists(f"{rd}timestamps.json") else {}
            rows.append(dict(r, cell=cid, attempt=att, pooled=pooled(rd), dir=rel(rd), measure_start=(ts.get("measure_start") or {}).get("wall_kst"), measure_end=(ts.get("measure_end") or {}).get("wall_kst"), flush=(ts.get("flush") or {}).get("body")))
    return rows

def perf_table(rows, title):
    L = [f"### {title}", "", "| cell | attempt | rep | workload | C | n | 성공/실패 | dur s | in tok | out tok | out tok/s | total tok/s | TTFT p50/p90/p95/p99 | TPOT p50/p90/p95/p99 | ITL p50/p95/p99 | E2EL p50/p95/p99 | pooled TTFT p95/p99 · TPOT p95/p99 | valid | measure_start | flush |", "|" + "---|" * 20]
    for r in rows:
        po = r.get("pooled") or {}
        L.append(f"| {r['cell']} | {r['attempt']} | {r['rep_id']} | {r['workload_id']} | {r['C']} | {r['n']} | {f1(r.get('completed'),0)}/{f1(r.get('failed'),0)} | {f1(r.get('duration'),1)} | {f1(r.get('input_tokens'),0)} | {f1(r.get('output_tokens'),0)} | {f1(r.get('output_tps'),2)} | {f1(r.get('total_tps'),1)} | {f1(r.get('ttft_p50'),0)}/{f1(r.get('ttft_p90'),0)}/{f1(r.get('ttft_p95'),0)}/{f1(r.get('ttft_p99'),0)} | {f1(r.get('tpot_p50'))}/{f1(r.get('tpot_p90'))}/{f1(r.get('tpot_p95'))}/{f1(r.get('tpot_p99'))} | {f1(r.get('itl_p50'))}/{f1(r.get('itl_p95'))}/{f1(r.get('itl_p99'))} | {f1(r.get('e2el_p50'),0)}/{f1(r.get('e2el_p95'),0)}/{f1(r.get('e2el_p99'),0)} | {f1(po.get('ttft_p95_pooled'),0)}/{f1(po.get('ttft_p99_pooled'),0)} · {f1(po.get('tpot_p95_pooled'))}/{f1(po.get('tpot_p99_pooled'))} | {r['valid']} | {r.get('measure_start')} | {str(r.get('flush'))[:40]} |")
    return L

def stats_table(rows, title):
    L = [f"### {title} (valid 반복만; sd = 표본표준편차 ddof=1; 표본 1 → sd null)", "", "| cell | workload | C | n_valid | out tok/s 원값 | 평균 | 중앙값 | 최소 | 최대 | sd | mean_of_rep TTFT p95 | mean_of_rep TPOT p95 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    g = {}
    for r in rows:
        if r["valid"]: g.setdefault((r["cell"], r["workload_id"], r["C"]), []).append(r)
    for (c, w, C), rs in g.items():
        v = [r["output_tps"] for r in rs]
        L.append(f"| {c} | {w} | {C} | {len(v)} | {', '.join(f'{x:.2f}' for x in v)} | {st.mean(v):.2f} | {st.median(v):.2f} | {min(v):.2f} | {max(v):.2f} | {sd(v)} | {st.mean(r['ttft_p95'] for r in rs):.0f} | {st.mean(r['tpot_p95'] for r in rs):.1f} |")
    return L

def main():
    C = cells(); state = json.load(open(f"{CAMP}/state/state.json")); env = json.load(open(f"{REPO}/shadow_assists/features/IDE_071/evidence/software_manifest.json"))
    events = [json.loads(l) for l in open(f"{CAMP}/state/execution_events.jsonl")]; branches = [l.strip() for l in open(f"{CAMP}/state/branch_trace.jsonl")] if os.path.exists(f"{CAMP}/state/branch_trace.jsonl") else []
    rows = rep_rows(C)
    # 원장 재계산 + 정합성
    u = {"PERF": 0, "REPLAY": 0, "LOAD": 0, "DIAG": 0, "RETRY": 0, "BOOT": 0, "GSM_Q": 0, "SESSIONS": 0, "NOT_SENT": 0}
    for e in events:
        if e["event"] == "session_end": u[e["kind"]] += 1; u["SESSIONS"] += 1
        if e["event"] == "boot_end": u["BOOT"] += 1
        if e["event"] == "gsm_end": u["GSM_Q"] += e["questions"]
        if e["event"] == "not_sent": u["NOT_SENT"] += 1
    n_gsm_files = len(glob.glob(f"{CAMP}/*/a*/gsm20_paired.json")); consistency = [("GSM 원장 문항 수 = 파일 문항 수", u["GSM_Q"] == 20 * n_gsm_files, f"{u['GSM_Q']} vs {20*n_gsm_files}"),
                                                                             ("부팅 원장 = status.json 수", u["BOOT"] == len(C), f"{u['BOOT']} vs {len(C)}"),
                                                                             ("PERF+LOAD+DIAG 원장 = valid/invalid 반복 행 수", u["PERF"] + u["LOAD"] + u["DIAG"] == len(rows), f"{u['PERF']+u['LOAD']+u['DIAG']} vs {len(rows)}")]
    # cells.jsonl
    with open(f"{FEAT}/manifests/cells.jsonl", "w") as f:
        for cid, att, adir, stt, sp in C:
            eff = json.load(open(f"{adir}effective_config.json")) if os.path.exists(f"{adir}effective_config.json") else {}; pr = json.load(open(f"{adir}runtime_proofs.json")) if os.path.exists(f"{adir}runtime_proofs.json") else {}
            f.write(json.dumps({"campaign_id": "IDE_072_20260916", "cell_id": cid, "attempt_id": att, "parent": sp.get("parent_cell_id"), "config_hash": sp.get("config_hash"), "requested": {"env": sp.get("env"), "server_args": sp.get("server_args"), "pin": sp.get("pin_nonkt"), "patch": sp.get("patch")},
                                "effective_server_args": eff.get("effective_server_args"), "server_args_log_graph_fields": pr.get("server_args_log_graph_fields"), "capture_log_lines": pr.get("capture_log_lines"), "per_layer_sum": pr.get("per_layer_sum"), "cf_ready_lines": pr.get("cf_ready_lines"),
                                "kt_worker_cpus": stt.get("kt_worker_cpus"), "pin": stt.get("pin"), "boot": stt.get("boot"), "status": stt.get("exit_status") or stt.get("state"), "replay": stt.get("replay"), "gsm20": stt.get("gsm20"), "dir": rel(adir), "code_sha": {"sglang": "71de97b2+local", "ktransformers": "6d460cc1+local", "kt_kernel_ext_sha256": env["software"]["kt_kernel_ext_sha256"].split()[0]}}, ensure_ascii=False) + "\n")
    # 요청별 표
    req_idx = []
    for r in rows:
        rq = f"{REPO}/{r['dir']}requests.jsonl"
        if not os.path.exists(rq): continue
        name = f"{r['cell']}__{r['attempt']}__{r['rep_id']}.md"; L = [f"# 요청별 — {r['cell']} {r['attempt']} {r['rep_id']}", "", f"원본 `{r['dir']}requests.jsonl`", "", "| idx | in | out | TTFT ms | TPOT ms (src) | ITL n | ITL sum | E2EL | error |", "|---|---|---|---|---|---|---|---|---|"]
        n = 0
        for l in open(rq):
            j = json.loads(l); n += 1; L.append(f"| {j['idx']} | {j['input_len']} | {j['output_len']} | {f1(j.get('ttft_ms'))} | {f1(j.get('tpot_ms'),2)} ({j.get('tpot_source')}) | {j.get('itl_count')} | {f1(j.get('itl_sum_ms'))} | {f1(j.get('e2el_ms'))} | {j.get('error') or ''} |")
        open(f"{FEAT}/reports/request_tables/{name}", "w").write("\n".join(L) + "\n"); req_idx.append((r["cell"], r["attempt"], r["rep_id"], n, f"reports/request_tables/{name}"))
    # 품질 paired
    Q = {}
    for cid, att, adir, stt, sp in C:
        g = f"{adir}gsm20_paired.json"
        if os.path.exists(g): Q[cid] = json.load(open(g))
    QL = ["# GSM20_PAIRED — 네 구성 × 동일 20문항 원기록 (IDE_072)", "", "선택: GSM8K test 행 0..39 (원본 GSM40 = gsm_eval.py head(40)) 을 index 정렬 후 앞 20 (0..19). greedy(temperature 0), max_tokens 1024, 동시성 4, 문항별 반복 없음. token_ids = null (OpenAI chat API 미제공). 과거 GSM40 (768 tokens·40문항) 과 분모·한도가 다름.", ""]
    for cid, j in Q.items(): QL.append(f"- {cid}: n {j['n']}, correct {j['correct']}, incorrect {j['incorrect']}, unscored {j['unscored']}, truncated {j['truncated']}, errors {j['errors']}; prompt_template_sha256 {j['prompt_template_sha256'][:16]}…, harness_sha256 {j['harness_sha256'][:16]}…; 파일 `{rel(f'{CAMP}/{cid}/a1/gsm20_paired.json')}`")
    QL += ["", "## 문항별 (구성별 추출답 · 정오 · finish_reason · completion_tokens)", "", "| idx | gold | " + " | ".join(f"{c}: pred/correct/finish/tokens" for c in Q) + " |", "|---|---|" + "---|" * len(Q)]
    if Q:
        first = next(iter(Q.values()))
        for i, qr in enumerate(first["results"]):
            cells_txt = []
            for c, j in Q.items():
                x = j["results"][i] if i < len(j["results"]) else {}
                cells_txt.append(f"{x.get('pred')} / {x.get('correct')} / {x.get('finish_reason')} / {x.get('completion_tokens')}")
            QL.append(f"| {qr['idx']} | {qr['gold']} | " + " | ".join(cells_txt) + " |")
        pairs = [("V0_confirm", "V1_confirm"), ("V0_confirm", "V2_first"), ("V2_first", "Q0_first")]
        QL += ["", "## 짝비교 (같은 문항의 출력 전문 일치 / 추출답 일치 — 기계적 비교만)", "", "| 쌍 | 출력 전문 일치 | 추출답 일치 | 정오 일치 | 비교 문항 수 |", "|---|---|---|---|---|"]
        for a, b in pairs:
            if a in Q and b in Q:
                A = {x["idx"]: x for x in Q[a]["results"]}; B = {x["idx"]: x for x in Q[b]["results"]}; ks = sorted(set(A) & set(B))
                QL.append(f"| {a}–{b} | {sum(1 for k in ks if A[k]['text'] == B[k]['text'])} | {sum(1 for k in ks if A[k]['pred'] == B[k]['pred'])} | {sum(1 for k in ks if A[k]['correct'] == B[k]['correct'])} | {len(ks)} |")
            else: QL.append(f"| {a}–{b} | — | — | — | 0 (미실행 구성) |")
        QL += ["", "## 출력 전문", ""]
        for c, j in Q.items():
            for x in j["results"]:
                QL += [f"### {c} · idx {x['idx']} (gold {x['gold']}, pred {x['pred']}, correct {x['correct']}, finish {x['finish_reason']}, tokens {x['completion_tokens']}, error {x['error']})", "", "```", (x["text"] or "").replace("```", "'''"), "```", ""]
    open(f"{FEAT}/quality/paired_question_results.md", "w").write("\n".join(QL) + "\n")
    # 실패 증거
    FL = ["# failures — IDE_072", "", "| cell | attempt | stage | 시각 | 원문 | 로그 |", "|---|---|---|---|---|---|"]
    for cid, att, adir, stt, sp in C:
        for e in stt.get("errors", []):
            txt = str(e.get("reason") or e.get("verdict") or e.get("msg") or e.get("exception") or e.get("error_lines"))[:300].replace("|", "\\|").replace("\n", " ")
            FL.append(f"| {cid} | {att} | {e.get('stage')} | {(e.get('t') or {}).get('wall_kst','')[:19]} | {txt} | {rel(adir)}server.full.log.gz |")
        rp = state.get("reproduction", {}).get(cid.split("_")[0])
        if rp and cid.endswith("_first"): FL.append(f"| {cid} | {att} | REPRODUCTION | — | state={rp.get('state')} nonfinite={json.dumps(rp.get('nonfinite'))[:300]} | — |")
    open(f"{FEAT}/failures/README.md", "w").write("\n".join(FL) + "\n")
    # artifact manifest
    n_art = 0; tot = 0
    with open(f"{FEAT}/manifests/artifact_manifest.jsonl", "w") as f:
        for p in sorted(glob.glob(f"{CAMP}/**/*", recursive=True)):
            if os.path.isdir(p): continue
            sz = os.path.getsize(p); tot += sz; n_art += 1
            f.write(json.dumps({"path": rel(p), "bytes": sz, "sha256": sha256(p), "published_in_git": sz <= 5 * 1024 * 1024, "location": "server violet-h100-016 " + rel(p)}) + "\n")
    with open(f"{FEAT}/manifests/SHA256SUMS", "w") as f:
        for l in open(f"{FEAT}/manifests/artifact_manifest.jsonl"): j = json.loads(l); f.write(f"{j['sha256']}  {j['path']}\n")
    # ---------------- FULL_REPORT ----------------
    L = ["# IDE_072 FULL_REPORT — 측정 사실 기록 (평가·해석 없음)", "", f"생성 {now()['wall_kst']}. campaign_id IDE_072_20260916. 지시서 cpu_offload_no_03_compact_verification. 브랜치 feat/cpu-offload-ide071.", "",
         "## 1. 식별·시각·해시", "", f"- 캠페인 시작 {state['t_start']['wall_kst']} · 종료 {state.get('t_end', {}).get('wall_kst')} · 상태 **{state.get('status')}**",
         f"- 입력 해시: 지시서 사본 {sha256(FEAT + '/cpu_offload_no_03_compact_verification_지시서.md')} · IDE_071 FULL_REPORT {sha256(REPO + '/shadow_assists/features/IDE_071/FULL_REPORT.md')}",
         f"- 코드: SGLang 71de97b2+local, ktransformers 6d460cc1+local, kt_kernel_ext.so {env['software']['kt_kernel_ext_sha256'].split()[0]}, torch 2.13.0+cu130 (IDE_071 evidence/software_manifest.json 재사용, 변경 없음)",
         "- hotmap_v2 / layer_budget_5952 SHA256: " + ", ".join(f"{k} {env['files'][k]['sha256']}" for k in ("hotmap_v2", "layer_budget_5952")), "",
         "## 2. 구성·실제 적용값", "", "PLAN_RESOLVED.md 의 구성표·S2 원본 대조·SOURCE_CONFIG_CONFLICT. 셀별 요청/실효값·capture 목록·62층 예산 합·affinity 는 manifests/cells.jsonl.", "",
         "| cell | attempt | parent | 상태 | boot s | config_mismatch | per_layer 합 | cf 행 | capture bs 목록 (로그) | ServerArgs graph 필드 | pin 이동 스레드 | kt 코어 | env | server_args |", "|" + "---|" * 14]
    for cid, att, adir, stt, sp in C:
        pr = json.load(open(f"{adir}runtime_proofs.json")) if os.path.exists(f"{adir}runtime_proofs.json") else {}
        cap = next((re.search(r"bs=(\[[0-9, ]*\])", l).group(1) for l in pr.get("capture_log_lines", []) if "bs=[" in l), "null")
        L.append(f"| {cid} | {att} | {sp.get('parent_cell_id')} | {stt.get('exit_status') or stt.get('state')} | {f1((stt.get('boot') or {}).get('boot_seconds'),0)} | {stt.get('config_mismatch')} | {pr.get('per_layer_sum')} | {pr.get('cf_ready_lines')} | {cap} | {pr.get('server_args_log_graph_fields')} | {(stt.get('pin') or {}).get('moved_threads')} | {min(stt.get('kt_worker_cpus') or [0])}–{max(stt.get('kt_worker_cpus') or [0])} ({len(stt.get('kt_worker_cpus') or [])}) | `{json.dumps(sp.get('env'), ensure_ascii=False)}` | `{json.dumps(sp.get('server_args'), ensure_ascii=False)}` |")
    L += ["", "### 실제 실행 명령", ""]
    for cid, att, adir, stt, sp in C:
        L.append(f"- `{cid}/{att}` launch: `{open(f'{adir}launch_cmd.sh').read().strip().splitlines()[-1]}`")
        for bc in sorted(glob.glob(f"{adir}*/bench_cmd.sh")): L.append(f"  - {bc.split('/')[-2]}: `{open(bc).read().strip().splitlines()[-1]}`")
    L += ["", "## 3. 실행량 (원장 execution_events.jsonl 재계산)", "", f"- PERF {u['PERF']}/11 · REPLAY {u['REPLAY']}/3 · LOAD {u['LOAD']}/1 · DIAG {u['DIAG']}/1 · RETRY {u['RETRY']}/2 · 부하 세션 {u['SESSIONS']}/18 · 부팅 {u['BOOT']}/10 · GSM 문항 {u['GSM_Q']}/80 · 미전송 워크로드 이벤트 {u['NOT_SENT']}",
          "- 정합성: " + "; ".join(f"{n}: {'OK' if ok else 'MISMATCH'} ({d})" for n, ok, d in consistency), "", "## 4. 성능 — 반복별 원값 (1차 / 확인 / ALT / LOAD / DIAG 별도 표)", ""]
    for title, flt in (("4.1 첫 부팅 1차 (SHORT_COLD)", lambda r: r["cell"].endswith("_first")), ("4.2 확인 부팅 3회 (SHORT_COLD)", lambda r: r["cell"].endswith("_confirm") and r["workload_id"] == "SHORT_COLD"), ("4.3 별도 입력 COMPACT_ALT", lambda r: r["workload_id"] == "COMPACT_ALT"), ("4.4 연속 부하 COMPACT_LOAD", lambda r: r["workload_id"] == "COMPACT_LOAD"), ("4.5 D0 진단 (일반 성능에 합산하지 않음)", lambda r: r["cell"].startswith("D0"))):
        sub = [r for r in rows if flt(r)]; L += perf_table(sub, title) + [""] + (stats_table(sub, title + " 통계") + [""] if sub else ["(없음)", ""])
    L += ["## 5. REPLAY (WARM_SERVER_REPLAY, 워밍업과 같은 생성 인자 C16·32)", "", "| cell | rc | 완료/실패 | out tok/s | healthy_after | 종류 | 입력 비고 |", "|---|---|---|---|---|---|---|"]
    for cid, att, adir, stt, sp in C:
        rp = stt.get("replay")
        if rp: L.append(f"| {cid} | {rp.get('rc')} | {rp.get('completed')}/{rp.get('failed')} | {f1(rp.get('output_tps'),2)} | {rp.get('healthy_after')} | {rp.get('kind')} | {rp.get('input_note')} |")
    L += ["", "## 6. 오류 재현 상태", "", "| 구성 | 첫 부팅 상태 | 재현 상태 | nonfinite 원문 카운트 | 최초 행 |", "|---|---|---|---|---|"]
    for v, rp in state.get("reproduction", {}).items():
        lg = f"{CAMP}/{v}_first/a1/server.full.log.gz"; cnt = {}
        if os.path.exists(lg):
            txt = gzip.open(lg, "rb").read().decode("utf-8", "replace")
            cnt = {k: len(re.findall(pat, txt)) for k, pat in (("probability tensor assert", r"probability tensor contains either"), ("illegal memory access", r"illegal memory access"), ("CUBLAS_STATUS_EXECUTION_FAILED", r"CUBLAS_STATUS_EXECUTION_FAILED"), ("nan(단어)", r"\bnan\b"), ("inf(단어)", r"\binf\b"), ("device-side assert", r"device-side assert"))}
        L.append(f"| {v} | {rp.get('exit_status')} | {rp.get('state')} | {json.dumps(cnt)} (state.json 의 초기 카운트는 부분문자열 기준이었음: {json.dumps((rp.get('nonfinite') or {}).get('counts'))}) | {(rp.get('nonfinite') or {}).get('first_line')} |")
    L += ["", "FIRST_NONFINITE_LOCATION_NOT_COLLECTED: 레이어·expert 단위 계측 없음 (지시서 §5.2).", "", "## 7. 정확성 (GSM20_PAIRED)", "", "`quality/paired_question_results.md` (문항별 전문·추출답·정오·짝비교). 요약:", ""]
    for cid, j in Q.items(): L.append(f"- {cid}: correct {j['correct']}/{j['n']}, unscored {j['unscored']}, truncated {j['truncated']}, errors {j['errors']}")
    L += ["", "greedy4 smoke (부팅당 1세트): 각 셀 `smoke_greedy4.json`.", "", "## 8. 조건부 분기 (branch_trace.jsonl)", "", "```"] + branches + ["```", "", "## 9. 오류·재시도 시간순", "", "failures/README.md 참조.", "", "## 10. 실행시간 내역", "", "| cell | attempt | boot s | warmup s | 측정 합 s | replay s | 셀 시작 | 셀 종료 |", "|---|---|---|---|---|---|---|---|"]
    for cid, att, adir, stt, sp in C:
        L.append(f"| {cid} | {att} | {f1((stt.get('boot') or {}).get('boot_seconds'),0)} | {f1((stt.get('warmup') or {}).get('wall_seconds'),0)} | {sum((r.get('wall_seconds') or 0) for r in (stt.get('reps') or [])):.0f} | {f1((stt.get('replay') or {}).get('wall_seconds'),0)} | {(stt.get('t_created') or {}).get('wall_kst','')[:19]} | {(stt.get('t_finished') or {}).get('wall_kst','')[:19]} |")
    L += ["", "## 11. 미수집 항목", "", "- 레이어·expert·step 계측: NOT_COLLECTED (새 계측 패치 0회). token_ids: null (API 미제공). GPU stream 활동·작업 완료 이벤트: NOT_COLLECTED.", "- 시계열 (cpu/gpu/memory_timeseries.csv, thread_affinity_start/end.json) 는 각 rep 디렉터리.", "",
          "## 12. 참고 표 — IDE_071 S2 (출처 있는 과거 값, 새 반복과 합산하지 않음)", "", "| 출처 | 값 |", "|---|---|", "| IDE_071 S2_b3_v2_nu5952__confirm SHORT_COLD C64 (3회) | 800.75, 811.33, 805.33 (평균 805.80, sd 5.31) |", "| IDE_071 P3_avx_rb_0 (B3 + KT_AVX_RB=0, SHORT_COLD C64 3회) | 774.08, 792.82, 794.50 (평균 787.13, sd 11.34) |", "| IDE_071 S2 load1024 a1 / a2 | a1 워밍업 중 inf/nan assert 종료, 1024/1024 실패 / a2 842.50 tok/s 1024 성공 |", "",
          "## 13. 원본 파일 인벤토리", "", f"- 파일 {n_art}, 총 {tot/1e9:.2f} GB, manifests/artifact_manifest.jsonl, SHA256SUMS. 5 MB 초과 파일은 서버 보존 (git 미게시).", "", "### 요청별 표 (전체 shard)", ""]
    for c, a, r, n, p in req_idx: L.append(f"- [{c} {a} {r} ({n} 행)]({p})")
    open(OUTF, "w").write("\n".join(L) + "\n")
    R = ["# IDE_072 RESULT — 수치표·실행 상태·출처만", "", "측정값만 기록한다. 평가·해석·의견은 포함하지 않는다.", "", f"상태 **{state.get('status')}** · 실행량 PERF {u['PERF']}/11 REPLAY {u['REPLAY']}/3 LOAD {u['LOAD']}/1 DIAG {u['DIAG']}/1 RETRY {u['RETRY']}/2 부팅 {u['BOOT']}/10 GSM {u['GSM_Q']}/80", "", "| cell | attempt | 상태 | boot s | mismatch | per_layer 합 | replay | gsm20 |", "|---|---|---|---|---|---|---|---|"]
    for cid, att, adir, stt, sp in C:
        pr = json.load(open(f"{adir}runtime_proofs.json")) if os.path.exists(f"{adir}runtime_proofs.json") else {}; rp = stt.get("replay") or {}
        R.append(f"| {cid} | {att} | {stt.get('exit_status') or stt.get('state')} | {f1((stt.get('boot') or {}).get('boot_seconds'),0)} | {stt.get('config_mismatch')} | {pr.get('per_layer_sum')} | {rp.get('completed')}/{rp.get('failed')} healthy={rp.get('healthy_after')} | {stt.get('gsm20')} |")
    R += [""] + stats_table([r for r in rows if not r["cell"].startswith("D0")], "성능 통계 (1차·확인·ALT·LOAD 는 cell/workload 별 행)") + [""] + perf_table(rows, "반복별 원값") + ["", "정확성: quality/paired_question_results.md · 실패: failures/README.md · 상세: FULL_REPORT.md"]
    open(OUTR, "w").write("\n".join(R) + "\n"); print(OUTF, len(L), "lines; cells", len(C), "reps", len(rows), "artifacts", n_art, "usage", u, "consistency", consistency)

if __name__ == "__main__":
    main()
