#!/usr/bin/env python3
"""IDE_071 — 측정 사실만 담은 FULL_REPORT.md / RESULT.md / reports/*.md / manifests/artifact_manifest.jsonl / cells.jsonl 생성.
평가·해석·권고 문장 없음. 표는 phase/cell/rep 순.
사용: render_data_report.py
"""
import glob, gzip, hashlib, json, os, statistics as st, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
CAMP = f"{REPO}/eval/results/IDE_071_20260916"; CD = f"{CAMP}/compact"
OUTF = f"{FEAT}/FULL_REPORT.md"; OUTR = f"{FEAT}/RESULT.md"; REP = f"{FEAT}/reports"; os.makedirs(REP, exist_ok=True); os.makedirs(f"{REP}/request_tables", exist_ok=True)

def f1(x, n=1): return "null" if x is None else f"{x:.{n}f}"
def sd(v): return f"{st.stdev(v):.2f}" if len(v) > 1 else "null"
def rel(p): return os.path.relpath(p, REPO)

def cell_dirs(root):
    out = []
    for cdir in sorted(glob.glob(f"{root}/*/")):
        cid = os.path.basename(cdir.rstrip("/"))
        if cid in ("manifests", "state", "compact", "specs"): continue
        for adir in sorted(glob.glob(f"{cdir}a*/")):
            sp = f"{adir}status.json"
            if os.path.exists(sp): out.append((cid, os.path.basename(adir.rstrip("/")), adir, json.load(open(sp))))
    return out

def spec_of(adir):
    p = f"{adir}cell_spec.json"; return json.load(open(p)) if os.path.exists(p) else {}

def boot_info(stt):
    b = stt.get("boot") or {}; return b.get("boot_seconds"), b.get("verdict")

def rep_rows(cid, att, adir, stt):
    rows = []
    for r in stt.get("reps") or []:
        rd = f"{adir}{r['rep_id']}/"; ts = json.load(open(f"{rd}timestamps.json")) if os.path.exists(f"{rd}timestamps.json") else {}
        pooled = None
        rq = f"{rd}requests.jsonl"
        if os.path.exists(rq):
            tt = []; tp = []; n = 0; outs = 0
            for l in open(rq):
                j = json.loads(l); n += 1; outs += (j.get("output_len") or 0)
                if j.get("ttft_ms") is not None: tt.append(j["ttft_ms"])
                if j.get("tpot_ms") is not None: tp.append(j["tpot_ms"])
            def p95(v): v = sorted(v); return v[int(round(0.95 * (len(v) - 1)))] if v else None
            pooled = {"n": n, "out_tokens_sum": outs, "ttft_p95_pooled": p95(tt), "tpot_p95_pooled": p95(tp)}
        rows.append({"cell": cid, "attempt": att, **r, "measure_start": (ts.get("measure_start") or {}).get("wall_kst"), "measure_end": (ts.get("measure_end") or {}).get("wall_kst"), "flush": (ts.get("flush") or {}).get("body"), "pooled": pooled, "dir": rel(rd)})
    return rows

def table(rows, title, wl_filter=None):
    L = [f"### {title}", "", "| cell | attempt | rep | workload | C | n | 성공/실패 | dur s | in tok | out tok | out tok/s | total tok/s | TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | ITL p50/p95 | E2EL p50/p95 | pooled TTFT p95 / TPOT p95 | valid | measure_start (KST) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if wl_filter and r["workload_id"] not in wl_filter: continue
        po = r.get("pooled") or {}
        L.append(f"| {r['cell']} | {r['attempt']} | {r['rep_id']} | {r['workload_id']} | {r['C']} | {r['n']} | {f1(r.get('completed'),0)}/{f1(r.get('failed'),0)} | {f1(r.get('duration'),1)} | {f1(r.get('input_tokens'),0)} | {f1(r.get('output_tokens'),0)} | {f1(r.get('output_tps'),2)} | {f1(r.get('total_tps'),1)} | {f1(r.get('ttft_p50'),0)}/{f1(r.get('ttft_p95'),0)}/{f1(r.get('ttft_p99'),0)} | {f1(r.get('tpot_p50'))}/{f1(r.get('tpot_p95'))}/{f1(r.get('tpot_p99'))} | {f1(r.get('itl_p50'))}/{f1(r.get('itl_p95'))} | {f1(r.get('e2el_p50'),0)}/{f1(r.get('e2el_p95'),0)} | {f1(po.get('ttft_p95_pooled'),0)} / {f1(po.get('tpot_p95_pooled'))} | {r['valid']} | {r.get('measure_start')} |")
    return L

def stats_table(rows, title):
    L = [f"### {title} — 반복 통계 (valid 반복만, sd = 표본표준편차 ddof=1)", "", "| cell | workload | C | n_valid | out tok/s: 값 | 평균 | 중앙값 | 최소 | 최대 | sd | TTFT p95 평균(mean_of_rep_p95) | TPOT p95 평균 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = {}
    for r in rows:
        if not r["valid"]: continue
        keys.setdefault((r["cell"], r["workload_id"], r["C"]), []).append(r)
    for (c, w, C), rs in keys.items():
        v = [r["output_tps"] for r in rs]
        L.append(f"| {c} | {w} | {C} | {len(v)} | {', '.join(f'{x:.2f}' for x in v)} | {st.mean(v):.2f} | {st.median(v):.2f} | {min(v):.2f} | {max(v):.2f} | {sd(v)} | {st.mean(r['ttft_p95'] for r in rs):.0f} | {st.mean(r['tpot_p95'] for r in rs):.1f} |")
    return L

def status_table(cells, title):
    L = [f"### {title}", "", "| cell | phase | parent | attempt | exit_status | boot s | 부팅 verdict | config_mismatch | errors | env | server_args (R0 대비 변경) | pin | patch | dir |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cid, att, adir, stt in cells:
        sp = spec_of(adir); bs, bv = boot_info(stt)
        L.append(f"| {cid} | {sp.get('phase_id')} | {sp.get('parent_cell_id')} | {att} | {stt.get('exit_status') or stt.get('state')} | {f1(bs,0)} | {bv} | {stt.get('config_mismatch')} | {len(stt.get('errors', []))} | `{json.dumps(sp.get('env', {}), ensure_ascii=False)}` | `{json.dumps(sp.get('server_args', {}), ensure_ascii=False)[:300]}` | {sp.get('pin_nonkt')} | {sp.get('patch')} | {rel(adir)} |")
    return L

def error_lines(cells):
    L = ["### 오류·재시도 시간순", "", "| 시각(KST) | cell | attempt | stage | 내용 | 로그 |", "|---|---|---|---|---|---|"]
    ev = []
    for cid, att, adir, stt in cells:
        for e in stt.get("errors", []):
            ev.append(((e.get("t") or {}).get("wall_kst", ""), cid, att, e.get("stage"), str(e.get("reason") or e.get("verdict") or e.get("msg") or e.get("exception") or e.get("error_lines"))[:300].replace("|", "\\|").replace("\n", " "), rel(adir) + "server.full.log.gz"))
    for e in sorted(ev): L.append("| " + " | ".join(str(x) for x in e) + " |")
    if len(ev) == 0: L.append("| — | — | — | — | 없음 | — |")
    return L

def write_request_tables(rows):
    idx = []
    for r in rows:
        rq = f"{REPO}/{r['dir']}requests.jsonl"
        if not os.path.exists(rq): continue
        name = f"{r['cell']}__{r['attempt']}__{r['rep_id']}.md"; p = f"{REP}/request_tables/{name}"
        L = [f"# 요청별 데이터 — {r['cell']} {r['attempt']} {r['rep_id']}", "", f"원본 `{r['dir']}requests.jsonl` (vllm bench `--save-detailed` 의 input_lens/output_lens/ttfts/itls 에서 도출; tpot = vllm tpots 또는 mean(ITL)).", "", "| idx | input_len | output_len | TTFT ms | TPOT ms (source) | ITL count | ITL sum ms | E2EL ms | error |", "|---|---|---|---|---|---|---|---|---|"]
        n = 0
        for l in open(rq):
            j = json.loads(l); n += 1
            L.append(f"| {j['idx']} | {j['input_len']} | {j['output_len']} | {f1(j.get('ttft_ms'))} | {f1(j.get('tpot_ms'),2)} ({j.get('tpot_source')}) | {j.get('itl_count')} | {f1(j.get('itl_sum_ms'))} | {f1(j.get('e2el_ms'))} | {j.get('error') or ''} |")
            if n % 2000 == 0: L.append(f"\n(분할: {n} 행까지)\n")
        open(p, "w").write("\n".join(L) + "\n"); idx.append((r["cell"], r["attempt"], r["rep_id"], n, f"reports/request_tables/{name}"))
    return idx

def artifact_manifest():
    out = f"{FEAT}/manifests/artifact_manifest.jsonl"; n = 0; total = 0
    with open(out, "w") as f:
        for root in (CAMP,):
            for p in sorted(glob.glob(f"{root}/**/*", recursive=True)):
                if os.path.isdir(p): continue
                sz = os.path.getsize(p); total += sz
                big = sz > 5 * 1024 * 1024
                f.write(json.dumps({"path": rel(p), "bytes": sz, "sha256": sha256(p), "published_in_git": not big and not p.endswith(".pt") and not p.endswith("perf.data"), "location": "server violet-h100-016 " + rel(p)}) + "\n"); n += 1
    with open(f"{FEAT}/manifests/SHA256SUMS", "w") as f:
        for l in open(out):
            j = json.loads(l); f.write(f"{j['sha256']}  {j['path']}\n")
    return n, total

def main():
    env = json.load(open(f"{FEAT}/evidence/software_manifest.json"))
    camp = json.load(open(f"{CAMP}/manifests/campaign.json")); state = json.load(open(f"{CAMP}/state/state.json"))
    cs = json.load(open(f"{CD}/compact_state.json")) if os.path.exists(f"{CD}/compact_state.json") else {}
    old = cell_dirs(CAMP); comp = cell_dirs(CD)
    old_rows = [r for c in old for r in rep_rows(*c)]; comp_rows = [r for c in comp for r in rep_rows(*c)]
    all_cells = old + comp
    # cells.jsonl
    with open(f"{FEAT}/manifests/cells.jsonl", "w") as f:
        for cid, att, adir, stt in all_cells:
            sp = spec_of(adir); eff = json.load(open(f"{adir}effective_config.json")) if os.path.exists(f"{adir}effective_config.json") else {}
            f.write(json.dumps({"campaign_id": camp["campaign_id"], "phase_id": sp.get("phase_id"), "cell_id": cid, "attempt_id": att, "parent_cell_id": sp.get("parent_cell_id"), "config_hash": eff.get("config_hash") or sp.get("config_hash"),
                                "code_sha": {"sglang": "71de97b264b04dcd514cf904003028aefe9775c8+local", "ktransformers": "6d460cc10780f2e0ad541b5d58ad28086dd32ef0+local", "kt_kernel_ext_sha256": "e29357f786a53030296f75beaf3db2745baa730835ddc7d98aca88dd014aa46d"},
                                "semantics_family": sp.get("semantics_family"), "precision_family": sp.get("precision_family"), "GPU_count": len(sp.get("gpus", "0,1,2,3").split(",")), "TP": (sp.get("server_args") or {}).get("tp", 4),
                                "requested": {"env": sp.get("env"), "server_args": sp.get("server_args"), "pin_nonkt": sp.get("pin_nonkt"), "patch": sp.get("patch")}, "effective_server_args": eff.get("effective_server_args"),
                                "boot_seconds": (stt.get("boot") or {}).get("boot_seconds"), "kt_worker_cpus": stt.get("kt_worker_cpus"), "pin": stt.get("pin"), "status": stt.get("exit_status") or stt.get("state"), "n_reps": len(stt.get("reps") or []), "gsm40": stt.get("gsm40"), "dir": rel(adir)}, ensure_ascii=False) + "\n")
    req_idx = write_request_tables(old_rows + comp_rows)
    n_art, tot_bytes = artifact_manifest()
    # ---------------- FULL_REPORT ----------------
    L = ["# IDE_071 FULL_REPORT — 측정 사실 기록 (평가·해석 없음)", "", f"생성 {now()['wall_kst']}. campaign_id {camp['campaign_id']}. 지시서: cpu_offload_no_02 (12단계) 로 시작 → 2026-09-16 07:05 부터 cpu_offload_no_02_compact 가 범위를 대체. 브랜치 feat/cpu-offload-ide071.", ""]
    L += ["## 1. 식별·시각·해시", "", f"- 캠페인 시작 {state['t_start']['wall_kst']} · compact 시작 {cs.get('t_start', {}).get('wall_kst')} · compact 종료 {cs.get('t_end', {}).get('wall_kst')} · compact 상태 {cs.get('status')}",
          f"- 지시서 SHA256: 12단계 {camp.get('instruction_sha256')} · compact {sha256(FEAT + '/cpu_offload_no_02_compact_지시서.md')}",
          f"- 입력 자료 해시: `evidence/software_manifest.json` files.* (IDE_069 FULL_REPORT, IDE_070 RESULT, hotmap v1/v2, layer_budget_5952, hotmap_mixed_0.25, 모델 config)",
          f"- 코드: SGLang git 71de97b2 + 로컬 수정 10 파일, ktransformers git 6d460cc1 + 로컬 수정 10 파일 (`evidence/source_changes.patch`), kt_kernel_ext.so sha256 e29357f7…, torch 2.13.0+cu130, flashinfer 0.6.17, triton 3.7.1, NCCL 2.29.7, vllm 벤치 0.26.1rc1.dev1177",
          f"- 하네스: `eval/ide071/` (git 커밋 참조 PUBLISH_RECEIPT.md)", ""]
    L += ["## 2. 하드웨어·소프트웨어·컨테이너·모델·hotmap 실제 구성", "", "```", env["host"]["hostname"], env["host"]["kernel"], env["host"]["os"], f"no_turbo={env['cpu']['no_turbo']} governor={env['cpu']['governor']} max_freq_khz={env['cpu']['max_freq_khz']}", env["cpu"]["turbostat_idle_5s"], env["memory"]["dimm_speed"], env["gpu"]["nvidia_smi"], "```",
          "- 컨테이너: " + "; ".join(f"{k}: {v['inspect']}" for k, v in env["containers"].items()), "- 모델: " + env["files"]["model_snapshot"][:300], "- CPU 가중치: " + env["files"]["kt_weights"].replace("\n", " | ")[:300],
          "- hotmap/예산 SHA256: " + ", ".join(f"{k} {v['sha256'][:16]}…" for k, v in env["files"].items() if k.startswith("hotmap") or k.startswith("layer")), "- kt 워커 코어 (측정): numa_0_t → cpu 0–47, numa_1_t → cpu 56–103 (모든 셀 status.json `kt_worker_cpus`)", ""]
    L += ["## 3. 등록된 모든 셀의 상태", "", "### 3.1 compact 범위 (cpu_offload_no_02_compact)", ""]
    L += status_table(comp, "compact 셀") + [""]
    L += ["재사용 대조군 (사용자 지시 '이미 수행한 것은 반복하지 않음'): " + json.dumps(cs.get("reused"), ensure_ascii=False), ""]
    L += ["### 3.2 12단계 계획에서 실행된 셀 (P1·P2·P3 일부; 07:05 이후 OUT_OF_SCOPE)", ""] + status_table(old, "12단계 셀") + [""]
    unrun = []
    for ph in ("P3", "P4", "P5", "P6", "P7"):
        p = f"{CAMP}/manifests/cells_{ph}.jsonl"
        if os.path.exists(p):
            ids = [json.loads(l)["cell_id"] for l in open(p)]; unrun += [(ph, i) for i in ids if i not in state["cells"]]
    L += [f"### 3.3 등록됐으나 미실행 (OUT_OF_SCOPE, compact 대체): {len(unrun)} 셀", "", ", ".join(f"{p}:{i}" for p, i in unrun), "", "P8~P11 은 manifest 생성 전 대체됨 (P8: W01~W07·T01~T03, P9: S01~S03 정의만 `eval/ide071/build_manifest.py`).", ""]
    L += ["## 4. 셀별 실제 실행 명령·환경변수 (launch_cmd.sh / bench_cmd.sh)", ""]
    for cid, att, adir, stt in all_cells:
        lc = open(f"{adir}launch_cmd.sh").read().strip().splitlines()[-1] if os.path.exists(f"{adir}launch_cmd.sh") else ""
        L.append(f"- `{cid}/{att}`: `{lc}`")
        bc = sorted(glob.glob(f"{adir}*/bench_cmd.sh"))
        if bc: L.append(f"  - bench 예: `{open(bc[0]).read().strip().splitlines()[-1][:400]}`")
    L += ["", "## 5. 모든 반복별 측정값", "", "### 5.1 compact", ""] + table(comp_rows, "compact 반복별") + [""] + stats_table(comp_rows, "compact") + ["", "### 5.2 12단계 (P1·P2·P3 실행분)", ""] + table(old_rows, "12단계 반복별") + [""] + stats_table(old_rows, "12단계") + [""]
    L += ["## 6. 계산 규칙", "", "```", "output_tps = vllm bench output_throughput = 수신 출력 token 수 / benchmark duration (첫 요청 제출 ~ 마지막 응답)", "TTFT/TPOT/ITL/E2EL p50/p95/p99 = vllm bench 요청 풀 백분위 (rep 단위); '평균' 열 = mean_of_rep_p95; pooled = requests.jsonl 재계산 (nearest-rank)", "TPOT(request) = vllm tpots 또는 mean(ITL); ITL = 연속 token 간격 (stream)", "표본표준편차 ddof=1; 1회 측정 sd = null; invalid 반복(요청 실패·서버 부재)은 통계에서 제외하고 원값은 표에 보존", "CPU busy = /proc/stat 1 s 코어별 (cpu_timeseries.csv: all/phys/smt/kt_cpus); GPU = nvidia-smi 2 s; RAM = free/numastat 5 s", "```", ""]
    L += ["## 7. 워크로드·precision·동시성·cache 정책 구분", "", "| workload | dataset | in/out | prefix | seed | ignore_eos | cache_policy | 요청 수 |", "|---|---|---|---|---|---|---|---|"]
    W = json.load(open(f"{REPO}/eval/ide071/configs/workloads.json"))
    for k, w in W.items(): L.append(f"| {k} | {w['dataset']} | {w.get('in')}/{w.get('out')} | {w.get('prefix')} | {w.get('seed')} | {w.get('ignore_eos')} | {w.get('cache_policy')} | n_per_C {w.get('n_per_C')} |")
    L += ["", "precision_family / semantics_family 는 §3 표와 manifests/cells.jsonl.", ""]
    L += ["## 8. 레이어·expert·step 데이터", "", "NOT_COLLECTED: compact 지시서 §6.2 (기존 하네스 범위 외, 계측 패치 미작성). 12단계 P6 CALIBRATION 트레이스는 미실행. 시계열: 각 rep 디렉터리 cpu_timeseries.csv / gpu_timeseries.csv / memory_timeseries.csv, thread_affinity_start/end.json.", ""]
    L += ["## 9. 정확성·출력 데이터", "", "| cell | attempt | greedy4 (prompt → 앞 60자) | GSM40 |", "|---|---|---|---|"]
    for cid, att, adir, stt in all_cells:
        g = stt.get("gsm40"); sm = stt.get("smoke")
        smt = "; ".join(f"{p}→{t}" for p, t in (sm or []))[:300].replace("|", "\\|").replace("\n", " ")
        L.append(f"| {cid} | {att} | {smt} | {g or 'null (미실행)'} |")
    L += ["", "GSM40 문항별 출력·정답·정오: 각 셀 `gsm40.json` (results[].gold/pred/ok/text_tail).", ""]
    L += ["## 10. 크래시·OOM·timeout·미지원·재시도 시간순", ""] + error_lines(all_cells) + [""]
    L += ["## 11. 선택 규칙 로그·종료 상태", "", "```"]
    if os.path.exists(f"{CD}/selection_trace.jsonl"): L += [l.strip() for l in open(f"{CD}/selection_trace.jsonl")]
    L += ["```", f"- compact 종료 상태: {cs.get('status')} · 실행량: {json.dumps(cs.get('budget'))}", f"- 12단계 selection_trace: `eval/results/IDE_071_20260916/state/selection_trace.jsonl` (P2 앵커 규칙 SEL-01~04)", ""]
    # 실행시간 내역
    L += ["## 12. 실제 전체 실행시간 내역 (부팅·준비 포함)", "", "| cell | attempt | boot s | warmup s | 측정 합 s | 셀 시작 (KST) | 셀 종료 (KST) |", "|---|---|---|---|---|---|---|"]
    for cid, att, adir, stt in all_cells:
        bs, _ = boot_info(stt); ws = (stt.get("warmup") or {}).get("wall_seconds"); ms = sum((r.get("wall_seconds") or 0) for r in (stt.get("reps") or []))
        L.append(f"| {cid} | {att} | {f1(bs,0)} | {f1(ws,0)} | {ms:.0f} | {(stt.get('t_created') or {}).get('wall_kst','')[:19]} | {(stt.get('t_finished') or {}).get('wall_kst','')[:19]} |")
    L += ["", "## 13. 원본 파일 인벤토리", "", f"- 파일 수 {n_art}, 총 {tot_bytes/1e9:.2f} GB, `manifests/artifact_manifest.jsonl` (path, bytes, sha256, published_in_git, location), `manifests/SHA256SUMS`.", "- 5 MB 초과 파일·.pt·perf.data 는 git 미게시 (서버 경로 참조).", "", "### 요청별 상세 표 (전체 shard 연결)", ""]
    for c, a, r, n, p in req_idx: L.append(f"- [{c} {a} {r} ({n} 행)]({p})")
    open(OUTF, "w").write("\n".join(L) + "\n")
    # ---------------- RESULT ----------------
    R = ["# IDE_071 RESULT — 수치표·실행 상태·출처만", "", "측정값만 기록한다. 평가·해석·의견은 포함하지 않는다.", "", "## compact (cpu_offload_no_02_compact)", ""] + status_table(comp, "compact 셀 상태") + [""] + stats_table(comp_rows, "compact") + [""] + table(comp_rows, "compact 반복별") + ["", f"재사용 대조군: {json.dumps(cs.get('reused'), ensure_ascii=False)}", f"실행량: {json.dumps(cs.get('budget'))} · 종료 상태 {cs.get('status')}", "", "## 12단계 계획 실행분 (P1·P2·P3 일부)", ""] + stats_table(old_rows, "12단계") + ["", "상세: FULL_REPORT.md §3~§13, manifests/cells.jsonl, reports/request_tables/."]
    open(OUTR, "w").write("\n".join(R) + "\n")
    print(OUTF, len(L), "lines;", OUTR, len(R), "lines; cells", len(all_cells), "reps", len(old_rows) + len(comp_rows), "artifacts", n_art)

if __name__ == "__main__":
    main()
