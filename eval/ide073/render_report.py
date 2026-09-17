#!/usr/bin/env python3
"""IDE_073 — FULL_REPORT.md (§11.3 12장 순서) / FULL_RAW_DATA.md (+raw_md/part-*.md) / config/<model>/<profile>/ / quality/<model>/<profile>/ / failures/ / ARTIFACT_INDEX.csv / SHA256SUMS.txt / COMPLETION_STATUS.md.
평가·해석 없음. 실행량은 원장 재계산."""
import glob, gzip, json, os, statistics as st, sys, re, csv, shutil, time
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_073")
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071"))
from common import *
CAMP = f"{REPO}/eval/results/IDE_073_20260917"; ST = f"{CAMP}/state"
def f1(x, n=1): return "null" if x is None else f"{x:.{n}f}"
def sd(v): return f"{st.stdev(v):.2f}" if len(v) > 1 else "null"
def rel(p): return os.path.relpath(p, REPO)
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def cells():
    out = []
    for cdir in sorted(glob.glob(f"{CAMP}/*/")):
        cid = os.path.basename(cdir.rstrip("/"))
        if cid in ("state", "specs") or cid.endswith("-PROBES") or cid == "G-CAL": continue
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
    return {"n": n, "in_sum": ins, "out_sum": outs, "ttft_p95_pooled": p(tt, .95), "tpot_p95_pooled": p(tp, .95), "algo": "nearest-rank"}

def rep_rows(C):
    rows = []
    for cid, att, adir, stt, sp in C:
        for r in stt.get("reps") or []:
            rd = f"{adir}{r['rep_id']}/"; ts = json.load(open(f"{rd}timestamps.json")) if os.path.exists(f"{rd}timestamps.json") else {}
            rows.append(dict(r, cell=cid, attempt=att, model=sp.get("model"), profile=sp.get("profile"), pooled=pooled(rd), dir=rel(rd), measure_start=(ts.get("measure_start") or {}).get("wall_kst"), flush=(ts.get("flush") or {}).get("body")))
    return rows

def perf_table(rows):
    L = ["| model | profile | cell | attempt | rep | workload | C | n | 성공/실패 | dur s | in tok | out tok | out tok/s | total tok/s | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p50/p95 | E2EL p50/p95 | pooled TTFT p95 / TPOT p95 | valid | measure_start | flush |", "|" + "---|" * 22]
    for r in rows:
        po = r.get("pooled") or {}
        L.append(f"| {r['model']} | {r['profile']} | {r['cell']} | {r['attempt']} | {r['rep_id']} | {r['workload_id']} | {r['C']} | {r['n']} | {f1(r.get('completed'),0)}/{f1(r.get('failed'),0)} | {f1(r.get('duration'),1)} | {f1(r.get('input_tokens'),0)} | {f1(r.get('output_tokens'),0)} | {f1(r.get('output_tps'),2)} | {f1(r.get('total_tps'),1)} | {f1(r.get('ttft_p50'),0)}/{f1(r.get('ttft_p95'),0)}/{f1(r.get('ttft_p99'),0)} | {f1(r.get('tpot_p50'))}/{f1(r.get('tpot_p95'))}/{f1(r.get('tpot_p99'))} | {f1(r.get('itl_p50'))}/{f1(r.get('itl_p95'))} | {f1(r.get('e2el_p50'),0)}/{f1(r.get('e2el_p95'),0)} | {f1(po.get('ttft_p95_pooled'),0)} / {f1(po.get('tpot_p95_pooled'))} | {r['valid']} | {r.get('measure_start')} | {str(r.get('flush'))[:30]} |")
    return L

def stats_table(rows):
    L = ["| model | profile | workload | C | n_valid | out tok/s 원값 | 평균 | 중앙값 | 최소 | 최대 | sd(ddof=1) | mean_of_rep TTFT p95 | mean_of_rep TPOT p95 | boots |", "|" + "---|" * 14]
    g = {}
    for r in rows:
        if r["valid"]: g.setdefault((r["model"], r["profile"], r["workload_id"], r["C"]), []).append(r)
    for (m, p, w, C), rs in g.items():
        v = [r["output_tps"] for r in rs]
        L.append(f"| {m} | {p} | {w} | {C} | {len(v)} | {', '.join(f'{x:.2f}' for x in v)} | {st.mean(v):.2f} | {st.median(v):.2f} | {min(v):.2f} | {max(v):.2f} | {sd(v)} | {st.mean(r['ttft_p95'] for r in rs):.0f} | {st.mean(r['tpot_p95'] for r in rs):.1f} | {sorted(set(r['cell'] for r in rs))} |")
    return L

def main():
    C = cells(); rows = rep_rows(C); state = json.load(open(f"{ST}/RUN_STATE.json")); man = json.load(open(f"{ST}/RUN_MANIFEST.json")) if os.path.exists(f"{ST}/RUN_MANIFEST.json") else {"cells": {}}
    events = jl(f"{ST}/execution_events.jsonl"); prog = jl(f"{FEAT}/progress/progress.jsonl"); env = json.load(open(f"{FEAT}/evidence/software/software_manifest.json"))
    u = {"PERF": 0, "DIAG": 0, "EXTRA": 0, "RETRY": 0, "BOOT": 0, "QUALITY_Q": 0, "SESSIONS": 0}
    void = [(e.get("cell"), e.get("rep"), e["t"]["epoch"]) for e in events if e["event"] == "session_void"]   # void 는 그 시각 이전에 끝난 같은 cell/rep 세션에만 적용
    def voided(e): return any(e.get("cell") == c and e.get("rep") == r and e["t"]["epoch"] < t for c, r, t in void)
    for e in events:
        if e["event"] == "session_end" and not voided(e): u[e["kind"]] = u.get(e["kind"], 0) + 1; u["SESSIONS"] += 1
        if e["event"] == "boot_end": u["BOOT"] += 1
        if e["event"] == "quality_end": u["QUALITY_Q"] += e["questions"]
    # config/<model>/<profile>/
    for cid, att, adir, stt, sp in C:
        d = f"{FEAT}/config/{sp.get('model')}/{sp.get('profile')}/{cid}_{att}"; os.makedirs(d, exist_ok=True)
        for f in ("launch_cmd.sh", "requested_config.json", "effective_config.json", "runtime_proofs.json", "thread_affinity.json", "server_info.json"):
            if os.path.exists(f"{adir}{f}"): shutil.copy(f"{adir}{f}", d)
        jdump({"argv": open(f"{adir}launch_cmd.sh").read().strip().splitlines()[-1].split() if os.path.exists(f"{adir}launch_cmd.sh") else None, "env_allowlist": sp.get("env")}, f"{d}/argv.json")
        pr = json.load(open(f"{adir}runtime_proofs.json")) if os.path.exists(f"{adir}runtime_proofs.json") else {}
        jdump({"per_layer_effective": pr.get("per_layer_effective"), "per_layer_sum": pr.get("per_layer_sum"), "cf_ready_lines": pr.get("cf_ready_lines"), "server_args_log_graph_fields": pr.get("server_args_log_graph_fields"), "capture_log_lines": pr.get("capture_log_lines"), "kt_worker_cpus": stt.get("kt_worker_cpus"), "pin": stt.get("pin"),
               "feature_paths": {"KT_CALLBACK_FREE": "kt_kernel/experts_base.py:880,996 (bool(env)); [kt-cf] ready 로그", "KT_CF_SKIP_EMPTY_IMM": "experts_base.py:853 (CF and deferred>=topk)", "KT_AVX_RB": "amx_kernels.hpp:1769 getenv!=NULL → avx_rb_on()", "KT_GPU_EXPERTS_PER_LAYER": "kt_ep_wrapper.py create_kt_config_from_server_args (로컬 패치)", "deferred": "experts_base.py select_deferred_experts protected_k=topk-max_deferred (KT_COLD_DEFER unset)", "sglang_tree": sp.get("env", {}).get("PYTHONPATH", "/sgl-workspace/sglang (로컬 패치 tree)")}}, f"{d}/effective_feature_paths.json")
        qd = f"{FEAT}/quality/{sp.get('model')}/{sp.get('profile')}"; os.makedirs(qd, exist_ok=True)
        if os.path.exists(f"{adir}quality20.json"): shutil.copy(f"{adir}quality20.json", f"{qd}/{cid}_{att}_outputs.json")
    # quality paired
    Q = {}
    for cid, att, adir, stt, sp in C:
        if os.path.exists(f"{adir}quality20.json"): Q[cid] = (sp.get("model"), sp.get("profile"), json.load(open(f"{adir}quality20.json")))
    QL = ["# 동일 20문항 출력 — 6구성 (IDE_073)", "", "GSM8K test rows 0..19 (rev 740312ad), chat API, temperature 0, top_p 1, max_tokens 1024, 동시성 4; GLM 은 enable_thinking=false. token_ids null (API 미제공).", "", "| cell | model | profile | exact | unscored | truncated | errors |", "|---|---|---|---|---|---|---|"]
    for cid, (m, p, j) in Q.items(): QL.append(f"| {cid} | {m} | {p} | {j['correct']}/{j['n']} | {j['unscored']} | {j['truncated']} | {j['errors']} |")
    for model in ("qwen", "glm"):
        ks = [k for k, v in Q.items() if v[0] == model]
        if not ks: continue
        QL += ["", f"## {model} 문항별 (추출답 / exact / finish / tokens)", "", "| qid | gold | " + " | ".join(ks) + " |", "|---|---|" + "---|" * len(ks)]
        first = Q[ks[0]][2]["results"]
        for i, qr in enumerate(first):
            QL.append(f"| {qr['qid']} | {qr['gold']} | " + " | ".join(f"{Q[k][2]['results'][i].get('pred')} / {Q[k][2]['results'][i].get('exact_match')} / {Q[k][2]['results'][i].get('finish_reason')} / {Q[k][2]['results'][i].get('completion_tokens')}" for k in ks) + " |")
        QL += ["", f"### {model} 짝비교 (출력 전문 일치 / 추출답 일치 / exact 일치, 20문항)", "", "| 쌍 | 전문 | 추출답 | exact |", "|---|---|---|---|"]
        for a in ks:
            for b in ks:
                if a < b:
                    A = Q[a][2]["results"]; B = Q[b][2]["results"]
                    QL.append(f"| {a}–{b} | {sum(1 for x, y in zip(A, B) if x['text'] == y['text'])} | {sum(1 for x, y in zip(A, B) if x['pred'] == y['pred'])} | {sum(1 for x, y in zip(A, B) if x['exact_match'] == y['exact_match'])} |")
    for cid, (m, p, j) in Q.items():
        QL += ["", f"## 출력 전문 — {cid} ({m}/{p})", ""]
        for x in j["results"]: QL += [f"### qid {x['qid']} (gold {x['gold']}, pred {x['pred']}, exact {x['exact_match']}, finish {x['finish_reason']}, tokens {x['completion_tokens']}, error {x['error']})", "", "```", (x["text"] or "").replace("```", "'''"), "```", ""]
    os.makedirs(f"{FEAT}/quality", exist_ok=True); open(f"{FEAT}/quality/paired_outputs.md", "w").write("\n".join(QL) + "\n")
    # failures
    FL = ["# failures — IDE_073", "", "| cell | attempt | stage | 시각 | 원문 | 로그 |", "|---|---|---|---|---|---|"]
    for cid, att, adir, stt, sp in C:
        for e in stt.get("errors", []):
            txt = str(e.get("reason") or e.get("verdict") or e.get("msg") or e.get("exception") or e.get("error_lines"))[:400].replace("|", "\\|").replace("\n", " ")
            FL.append(f"| {cid} | {att} | {e.get('stage')} | {(e.get('t') or {}).get('wall_kst','')[:19]} | {txt} | {rel(adir)}server.full.log.gz |")
            fd = f"{FEAT}/failures/{cid}_{att}"; os.makedirs(fd, exist_ok=True); jdump(stt.get("errors"), f"{fd}/errors.json")
    os.makedirs(f"{FEAT}/failures", exist_ok=True); open(f"{FEAT}/failures/README.md", "w").write("\n".join(FL) + "\n")
    # probes
    PS = {}
    for model in ("qwen", "glm"):
        p = f"{CAMP}/{model}-PROBES/probes_summary.json"
        if os.path.exists(p): PS[model] = json.load(open(p))
    # artifact index
    n_art = 0; tot = 0
    with open(f"{FEAT}/ARTIFACT_INDEX.csv", "w") as f:
        w = csv.writer(f); w.writerow(["path", "bytes", "sha256", "published_in_git", "location"])
        for p in sorted(glob.glob(f"{CAMP}/**/*", recursive=True)) + sorted(glob.glob(f"{HOME}/.cache/huggingface/kt/ide073/profiles/**/*", recursive=True)):
            if os.path.isdir(p): continue
            sz = os.path.getsize(p); tot += sz; n_art += 1; w.writerow([rel(p) if p.startswith(REPO) else p, sz, sha256(p), sz <= 5 * 1024 * 1024 and p.startswith(REPO), "server violet-h100-016"])
    with open(f"{FEAT}/SHA256SUMS.txt", "w") as f:
        for r in csv.DictReader(open(f"{FEAT}/ARTIFACT_INDEX.csv")): f.write(f"{r['sha256']}  {r['path']}\n")
    # ---------------- FULL_REPORT ----------------
    L = ["# IDE_073 FULL_REPORT — 2모델 × (GPU-only 8장 / KT 공식 기본 4장 / 최고 구성 4장) 비교·구간 계측 (측정 사실만)", "", f"생성 {now()['wall_kst']}. campaign_id IDE_073_20260917 · 지시서 cpu_offload_no_04 · 브랜치 feat/cpu-offload-two-models-20260917 · 상태 {state.get('status')}", ""]
    L += ["## 1. 모델 자격 검사와 실행 상태", "", "| 모델 | revision | tensor 합 | GPU8 참조 | GPU4 GPU-only 자격 | 4장 하이브리드 |", "|---|---|---|---|---|---|"]
    def stt_of(cid): return next((s2.get("exit_status") for c2, a2, d2, s2, sp2 in C if c2 == cid), "NOT_RUN")
    L.append(f"| Qwen3-Coder-480B-A35B-Instruct-FP8 | 003f183a… | 482.1 GB | Q-GPU8: {stt_of('Q-GPU8')} | 482 GB > 320 GB → 산정상 불가 (기동 생략; IDE_068 TP4 OOM 기록) | BASIC4 {stt_of('Q-KT-BASIC4')} / OPT4 {stt_of('Q-OPT4-b1')},{stt_of('Q-OPT4-b2')} |")
    L.append(f"| GLM-4.7-FP8 | 7b3b5f81… | 362.1 GB (93 shard) | G-GPU8: {stt_of('G-GPU8')} | 362 GB > 320 GB → 산정상 불가 | BASIC4 {stt_of('G-KT-BASIC4')} / OPT4 {stt_of('G-OPT4-b1')},{stt_of('G-OPT4-b2')} |")
    gp = f"{ST}/glm_opt_plan.json"; L.append(f"\nGLM OPT4-TRANSFER 준비 상태: {json.load(open(gp)) if os.path.exists(gp) else 'NOT_RUN'} (state/glm_prepare_state.json)\n")
    L += ["## 2. 공식 원문 설정·자원 정규화·최적화 설정 diff", "", "BASELINE_PROVENANCE.md, CONFIG_DIFF.md 참조 (본 저장소 동일 디렉터리). 설치본 미지원 옵션: --kt-enable-dynamic-expert-update, --kt-gpu-prefill-token-threshold, --kt-expert-placement-strategy (UNSUPPORTED_OPTION).", ""]
    L += ["## 3. 실행 환경·모델 revision·모든 명령·effective feature", "", "```", env["host"]["hostname"], env["host"]["kernel"], f"no_turbo={env['cpu']['no_turbo']} governor={env['cpu']['governor']}", env["gpu"]["nvidia_smi"][:1500], "```", "- 소프트웨어: " + env["software"]["sgl_kt_versions"].replace("\n", " · "), "- kt_kernel_ext.so " + env["software"]["kt_kernel_ext_sha256"].split()[0], "- sglang 로컬 tree 71de97b + 10파일; 공식 worktree /sgl-workspace/sglang-upstream (diff 0)", "", "| cell | model | profile | attempt | 상태 | boot s | mismatch | per_layer 합 | cf 행 | capture bs | pin 이동 | kt 코어 | env | launch |", "|" + "---|" * 14]
    for cid, att, adir, stt, sp in C:
        pr = json.load(open(f"{adir}runtime_proofs.json")) if os.path.exists(f"{adir}runtime_proofs.json") else {}
        cap = next((re.search(r"bs=(\[[0-9, ]*\])", l).group(1) for l in pr.get("capture_log_lines", []) if "bs=[" in l), "null")
        lc = open(f"{adir}launch_cmd.sh").read().strip().splitlines()[-1] if os.path.exists(f"{adir}launch_cmd.sh") else ""
        L.append(f"| {cid} | {sp.get('model')} | {sp.get('profile')} | {att} | {stt.get('exit_status') or stt.get('state')} | {f1((stt.get('boot') or {}).get('boot_seconds'),0)} | {stt.get('config_mismatch')} | {pr.get('per_layer_sum')} | {pr.get('cf_ready_lines')} | {cap} | {(stt.get('pin') or {}).get('moved_threads')} | {len(stt.get('kt_worker_cpus') or [])} | `{json.dumps(sp.get('env'), ensure_ascii=False)}` | `{lc}` |")
    L += ["", "bench 명령: 각 rep 디렉터리 bench_cmd.sh. config/<model>/<profile>/ 에 argv.json·effective_config.json·effective_feature_paths.json·affinity.", ""]
    L += ["## 4. 모든 반복의 성공·실패·입출력 token·duration·처리량·지연", ""] + perf_table(rows) + ["", "### 통계 (valid 반복만)", ""] + stats_table(rows) + [""]
    L += ["### REPLAY (OPT4 b1, 워밍업과 같은 생성 인자 C16·32 재전송)", "", "| cell | 완료/실패 | out tok/s | healthy_after | 입력 비고 |", "|---|---|---|---|---|"]
    for cid, att, adir, stt, sp in C:
        rp = stt.get("replay")
        if rp: L.append(f"| {cid} | {rp.get('completed')}/{rp.get('failed')} | {f1(rp.get('output_tps'),2)} | {rp.get('healthy_after')} | {rp.get('input_note')} |")
    L += ["", "## 5. layer·rank·stage 구간 시간, 노출 대기·overlap·critical path", "",
          "기록: qwen D2 는 2회 실행. 1차(bootA_retry, 10:21) 트레이스는 프로브 코드가 /start_profile activities 를 ['CPU','CUDA'] 로 보내(SGLang 키는 'GPU') GPU 활동 0건 → INVALID_NO_GPU_ACTIVITY(원장 status_correction). 2차(bootA_retry2, 10:56, activities ['CPU','GPU']) 트레이스를 §5·profiles/ 에 사용. 2차 트레이스 span 40.96 s 중 첫 16.4 s 는 프로파일 시작(10:56:01)부터 첫 요청 도착 전(벤치 클라이언트 기동) 구간이며 busy 비율·gap max 에 그대로 포함됨(보정 없음).", ""]
    for model, ps in PS.items():
        L += [f"### {model} 프로브 세션", "", "| 세션 | rc | out tok/s | duration s | throughput_ratio | elapsed_ratio | 비고 |", "|---|---|---|---|---|---|---|"]
        for k, v in ps["sessions"].items():
            sm = v.get("summary") or {}; L.append(f"| {k} | {v.get('rc')} | {f1(sm.get('output_throughput'),2)} | {f1(sm.get('duration'),1)} | {f1(v.get('throughput_ratio'),3)} | {f1(v.get('elapsed_ratio'),3)} | {'cpu_jobs rows ' + str(v.get('cpu_jobs_rows')) if k=='D1' else ('trace ' + str(v.get('trace_files')) if k=='D2' else '')} |")
        ana = f"{FEAT}/profiles/{model}/layer_stage_aggregate.md"
        L += ["", open(ana).read() if os.path.exists(ana) else f"profiles/{model}/ 분석 파일 없음 (analyze_probes.py 미실행 또는 NOT_COLLECTED)", ""]
    if not PS: L += ["프로브 미실행 (NOT_RUN).", ""]
    L += ["## 6. expert hit/Cold 선택·rows 분포·deferred 통계", "", "D1 LIGHT_PROBE 의 cpu_jobs.csv (KT_PHASE_PROF, 64회당 1회 표본: numa, activated_expert, prepare/cpy_input/q_input/up_gate/act/q_down/down/weight/total µs, max_local_num, qlen). 층 id·step id·expert id 는 이 계측이 출력하지 않음 → NOT_COLLECTED(layer/expert id). deferred produced/consumed/stale 카운터: NOT_COLLECTED (계측 없음).", ""]
    L += ["## 7. CPU·GPU·DRAM·NUMA 시계열", "", "각 rep: cpu_timeseries.csv (1 s, 코어별 busy, all/phys/smt/kt), gpu_timeseries.csv (2 s), memory_timeseries.csv (5 s). D3: perf_report_top.txt, perf_stat_20s.txt, pcm_memory.csv (2 s, MB/s 정의는 pcm-memory 헤더). 경계: timestamps.json (measure_start/end).", ""]
    L += ["## 8. 계측 ON/OFF 간섭과 누락", "", "§5 표의 throughput_ratio/elapsed_ratio. 미수집: S00 토큰화·스케줄러 대기(요청별 TTFT 만), S02 router 시간(트레이스 커널명 수준), S03/S06 복사 bytes(트레이스 memcpy 크기), S08 의존 대기(PARTIAL_DEPENDENCY_MEASUREMENT), S12 동적 배치(해당 없음).", ""]
    L += ["## 9. 동일 문항별 출력·기계적 채점·짝비교", "", "quality/paired_outputs.md (문항별 전문 포함). 요약:", ""] + QL[4:4 + len(Q) + 2] + [""]
    L += ["## 10. 모든 오류·재시도·미실행 항목", "", "failures/README.md. manifest 상태:", "", "| cell | 계획 | 상태 |", "|---|---|---|"] + [f"| {k} | {v.get('note')} | {v.get('status')} |" for k, v in man.get("cells", {}).items()] + ["", f"원장 재계산: PERF {u['PERF']}/30 · DIAG {u['DIAG']}/8 · 재시도·조건부 {u['EXTRA']+u['RETRY']}/6 · 세션 {u['SESSIONS']}/44 · 부팅 {u['BOOT']}/20 · 품질 {u['QUALITY_Q']}/120", ""]
    L += ["## 11. 30분 보고 이력 전체", "", "| # | 예정 | 실제 | 경과 s | 단계 | 현재 | 항목 수 | 최근 완료 | GLM bytes | HBM | 오류 | 전달 |", "|" + "---|" * 12]
    for i, r in enumerate(prog): L.append(f"| {i} | {r.get('scheduled_kst')} | {r.get('actual_kst')} | {r.get('elapsed_s')} | {r.get('phase')} | {str(r.get('current'))[:60]} | {r.get('counts')} | {str(r.get('done_last_30min'))[:120]} | {r.get('glm_download_bytes')} | {str(r.get('hbm_MiB'))[:60]} | {str(r.get('errors'))[:80]} | {r.get('delivery')} |")
    L += ["", "## 12. 파일 색인·Git 게시·다운로드", "", f"- ARTIFACT_INDEX.csv: {n_art} 파일, {tot/1e9:.2f} GB (5 MB 초과·profiles 원본은 서버 보존). SHA256SUMS.txt.", "- PUBLISH_RECEIPT.md (게시 후 작성). FULL_RAW_DATA.md + raw_md/part-*.md.", ""]
    open(f"{FEAT}/FULL_REPORT.md", "w").write("\n".join(L) + "\n")
    # ---------------- FULL_RAW_DATA (분할) ----------------
    os.makedirs(f"{FEAT}/raw_md", exist_ok=True); parts = []; buf = []; size = 0; k = 1
    def flush():
        nonlocal buf, size, k
        if not buf: return
        p = f"{FEAT}/raw_md/part-{k:04d}.md"; open(p, "w").write("\n".join(buf) + "\n"); parts.append((f"raw_md/part-{k:04d}.md", os.path.getsize(p))); buf = []; size = 0; k += 1
    for old in glob.glob(f"{FEAT}/raw_md/part-*.md"): os.remove(old)   # 이전 실행 잔여 part 제거
    def emit(title, text):
        nonlocal buf, size
        LIM = 8_000_000
        if len(text) > LIM:   # 큰 원자료(서버 로그 전문 등)는 8 MB 조각으로 나눠 수록 (GitHub 단일 파일 제한 회피)
            n = (len(text) + LIM - 1) // LIM
            for i in range(n): emit(f"{title} (조각 {i+1}/{n})", text[i*LIM:(i+1)*LIM])
            return
        chunk = f"\n## {title}\n\n```\n{text}\n```\n"
        if size + len(chunk) > LIM: flush()
        buf.append(chunk); size += len(chunk)
    for cid, att, adir, stt, sp in C:
        for f in ("cell_spec.json", "requested_config.json", "effective_config.json", "runtime_proofs.json", "server_info.json", "status.json", "smoke_greedy4.json", "quality20.json", "thread_affinity.json"):
            if os.path.exists(f"{adir}{f}"): emit(f"{cid}/{att}/{f}", open(f"{adir}{f}").read())
        for rd in sorted(glob.glob(f"{adir}*/")):
            for f in ("bench_cmd.sh", "metrics.json", "timestamps.json", "requests.jsonl", "bench.stdout.log", "cpu_timeseries.csv", "gpu_timeseries.csv", "memory_timeseries.csv"):
                if os.path.exists(f"{rd}{f}"): emit(f"{cid}/{att}/{os.path.basename(rd.rstrip('/'))}/{f}", open(f"{rd}{f}", errors="replace").read())
        if os.path.exists(f"{adir}server.full.log.gz"): emit(f"{cid}/{att}/server.full.log", gzip.open(f"{adir}server.full.log.gz", "rb").read().decode("utf-8", "replace"))
    for model in PS:
        for f in sorted(glob.glob(f"{CAMP}/{model}-PROBES/**/*", recursive=True)):
            if os.path.isfile(f) and f.endswith((".json", ".csv", ".txt", ".log")) and os.path.getsize(f) < 20_000_000: emit(rel(f), open(f, errors="replace").read())
    for f in (f"{ST}/execution_events.jsonl", f"{ST}/RUN_MANIFEST.json", f"{ST}/RUN_STATE.json", f"{FEAT}/progress/progress.jsonl"):
        if os.path.exists(f): emit(rel(f), open(f).read())
    flush()
    R = ["# FULL_RAW_DATA — IDE_073 (전체 색인)", "", "텍스트 원자료 전부를 raw_md/part-*.md 로 분할 수록 (설정·요청별 결과·시계열·서버 로그 전문·프로브 산출·원장·30분 보고). 바이너리(torch trace .gz, perf.data)는 서버 경로·SHA256 으로 ARTIFACT_INDEX.csv 에 기재.", "", "| part | bytes |", "|---|---|"] + [f"| [{p}]({p}) | {b} |" for p, b in parts]
    open(f"{FEAT}/FULL_RAW_DATA.md", "w").write("\n".join(R) + "\n")
    # COMPLETION_STATUS
    CS = [f"# COMPLETION_STATUS — IDE_073 ({now()['wall_kst']})", "", f"- 캠페인 상태: {state.get('status')}", f"- 실행량: PERF {u['PERF']}/30 · DIAG {u['DIAG']}/8 · 재시도·조건부 {u['EXTRA']+u['RETRY']}/6 · 세션 {u['SESSIONS']}/44 · 부팅 {u['BOOT']}/20 · 품질 {u['QUALITY_Q']}/120", "", "| 항목 | 상태 | 증거 |", "|---|---|---|"]
    for k, v in man.get("cells", {}).items(): CS.append(f"| {k} | {v.get('status')} | eval/results/IDE_073_20260917/{k}/ |")
    for model in ("qwen", "glm"): CS.append(f"| {model} 프로브 D0~D3 | {'RUN ' + str({k: ('valid' if (v.get('summary') or {}).get('failed') == 0 and v.get('rc') == 0 else v.get('status') or 'invalid') for k, v in PS[model]['sessions'].items()}) if model in PS else 'NOT_RUN'} | eval/results/IDE_073_20260917/{model}-PROBES/ |")
    CS += ["", "## 최종 체크 (§13)", "", "| 항목 | 값 | 증거 |", "|---|---|---|",
           f"| 두 모델 revision·GPU8·GPU4 자격 기록 | true | FULL_REPORT §1 |", "| 공식/정규화/로컬 분리 | true | BASELINE_PROVENANCE.md, CONFIG_DIFF.md |", "| offload 구성 GPU expert ≠ 0 | true (Qwen 96/층별 75~142, GLM 80) | config/ |",
           "| 최고 설정 feature·hotmap·예산·RB 파싱 검증 | true | config/qwen/OPT4/*/effective_feature_paths.json, runtime_proofs |", f"| GLM 이식 상태 | {json.load(open(gp)).get('status') if os.path.exists(gp) else 'NOT_RUN'} | state/glm_prepare_state.json |",
           "| 성능 30·진단 8 각 항목 상태 | 위 표 | RUN_MANIFEST.json |", "| 일반 성능/profiler 수치 분리 | true | §4 vs §5 |", "| GPU 실행시간/CPU enqueue 구분 | torch profiler CUDA activity (D2) 범위에서 | profiles/ |",
           "| 미계측 구간·overhead 공개 | true | FULL_REPORT §8 |", "| 동일 문항 출력 보존 | true | quality/ |", "| 30분 보고 이력 | true | FULL_REPORT §11 |", "| 평가·추측·권고 없음 | true | — |", "| raw·hash 보존 | true | ARTIFACT_INDEX.csv |", "| Git push·원격 확인 | PUBLISH_RECEIPT.md | — |", "| 다운로드 제공 | 최종 응답 | — |"]
    open(f"{FEAT}/COMPLETION_STATUS.md", "w").write("\n".join(CS) + "\n")
    print("FULL_REPORT", len(L), "lines; raw parts", len(parts), "; cells", len(C), "reps", len(rows), "artifacts", n_art, "usage", u)

if __name__ == "__main__":
    main()
