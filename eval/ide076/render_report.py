#!/usr/bin/env python3
"""IDE_076 — FULL_REPORT.md (원값·유효성·파일), COMPLETION_STATUS.md, ARTIFACT_INDEX.csv, SHA256SUMS.txt 생성 (PLAN.md §16.4: 판단 문서는 별도 수동).
입력: compare_variants.py 산출(e2e_comparison.csv, e2e_stats.json), replay/*/expert_results.csv, A1_BRANCH_PROOF.json, state/*.log, chain.log, execution_events.jsonl"""
import json, os, csv, glob, hashlib, time, statistics as st
HOME = os.path.expanduser("~"); REPO = f"{HOME}/projects/vllm_hybrid"; FEAT = f"{REPO}/shadow_assists/features/IDE_076"; CAMP = f"{REPO}/eval/results/IDE_076_20260917"; ST = f"{CAMP}/state"


def sha_plain(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def f1(x, n=1): return "null" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def main():
    now = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")
    os.system(f"cd {REPO} && python3 eval/ide076/compare_variants.py > /dev/null 2>&1")
    rows = list(csv.DictReader(open(f"{FEAT}/e2e_comparison.csv"))); stats = json.load(open(f"{FEAT}/e2e_stats.json"))
    L = [f"# FULL_REPORT — IDE_076 A1·A2·B 구현·검증 ({now})", "", "원값 보고서. 예상 speedup·우수성 주장·다음 제안은 없음 (판단은 ANALYSIS_AND_DECISION.md, *_DECISION.md).", "",
         "## 1. 환경·실효 설정", "", "BASELINE_PROVENANCE.md, SOURCE_MANIFEST.json, PLAN_RESOLVED.md 참조. 바이너리: v4 12926df2… (reference), v5 941a83c4… (A1 런타임 분기), v6 194f3064… (v5 + A2 probe), v7 (A1 템플릿·hoist + 스레드별 카운터; 아래 부팅 표의 .so 열).", "",
         "## 2. E2E 세션 원값 (OFF = 무계측 성능; vllm bench / vllmw)", "", "| variant | boot | session | .so | A1 | A2 | hotmap | workload | mode | client | done/sent | in/out tok | dur s | tps | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p95 | E2EL p95 | valid | t_start |", "|" + "---|" * 20]
    for r in rows: L.append(f"| {r['variant_id']} | {r['boot_id']} | {r['session_id']} | {r['binary_sha256'][:8]} | {r['a1']} | {r['a2']} | {os.path.basename(r['hotmap'] or '')} | {r['workload_id'].replace('M073_QWEN_','').replace('M076_QWEN_','')} | {r['mode']} | {r['client']} | {r['completed']}/{r['sent']} | {r['input_tokens']}/{r['output_tokens']} | {f1(float(r['duration_s']) if r['duration_s'] else None)} | {f1(float(r['output_tps']) if r['output_tps'] else None,2)} | {f1(float(r['ttft_p50_ms']) if r['ttft_p50_ms'] else None,0)}/{f1(float(r['ttft_p95_ms']) if r['ttft_p95_ms'] else None,0)}/{f1(float(r['ttft_p99_ms']) if r['ttft_p99_ms'] else None,0)} | {f1(float(r['tpot_p50_ms']) if r['tpot_p50_ms'] else None)}/{f1(float(r['tpot_p95_ms']) if r['tpot_p95_ms'] else None)}/{f1(float(r['tpot_p99_ms']) if r['tpot_p99_ms'] else None)} | {f1(float(r['itl_p95_ms']) if r['itl_p95_ms'] else None)} | {f1(float(r['e2el_p95_ms']) if r['e2el_p95_ms'] else None,0)} | {r['execution_valid']} | {r['t_start'][11:19]} |")
    L += ["", "## 3. variant × workload 통계 (유효 세션만; sd ddof=1; pooled = Σtok/Σdur)", "", "| variant | workload | mode | n | boots | mean | median | sd | min | max | pooled | TTFT p95 mean | TPOT p95 mean | invalid |", "|" + "---|" * 14]
    for k, v in stats["stats"].items(): L.append(f"| {v['variant']} [{v.get('population','')}] | {v['workload'].replace('M073_QWEN_','').replace('M076_QWEN_','')} | {v['mode']} | {v['n_sessions']} | {v['n_boots']} | {f1(v['tps_mean'],2)} | {f1(v['tps_median'],2)} | {f1(v['tps_sd_ddof1'],2)} | {f1(v['tps_min'],1)} | {f1(v['tps_max'],1)} | {f1(v['tps_pooled'],2)} | {f1(v['ttft_p95_mean'],0)} | {f1(v['tpot_p95_mean'],1)} | {v['invalid_sessions']} |")
    L += ["", "R0 대비 이득 (MAIN OFF; 총 개선 = 평균비 − 1, 짝 = 부팅 순서 짝; 퍼센트 합산 없음):", "", "```", json.dumps(stats["gains_vs_R0"], indent=1, ensure_ascii=False), "```", ""]
    L += ["## 4. expert replay 원값 (컨테이너 CPU 전용, 난수 가중치, p50/p10/p90 µs; replay/<variant>/expert_results.csv)", "", "| variant | scenario | n | n_unique | rows/expert | p50 | p10 | p90 | min |", "|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(glob.glob(f"{CAMP}/replay/*/expert_results.csv")):
        rr = list(csv.DictReader(open(d))); v = d.split("/")[-2]
        for sc in sorted(set(r["scenario"] for r in rr)):
            t = sorted(int(r["elapsed_ns"]) for r in rr if r["scenario"] == sc); x = [r for r in rr if r["scenario"] == sc][0]
            L.append(f"| {v} | {sc} | {len(t)} | {x['n_unique']} | {x['rows_per_expert']} | {t[len(t)//2]/1e3:.1f} | {t[len(t)//10]/1e3:.1f} | {t[int(len(t)*.9)]/1e3:.1f} | {t[0]/1e3:.1f} |")
    if os.path.exists(f"{FEAT}/A1_BRANCH_PROOF.json"): L += ["", "## 5. feature proof", "", "```", open(f"{FEAT}/A1_BRANCH_PROOF.json").read()[:3000], "```", ""]
    for lg in ("a1_flag_test.log", "a1_fp8_regression.log", "v6_fp8_regression.log", "v7_fp8_regression.log"):
        p = f"{ST}/{lg}"
        if os.path.exists(p): L += [f"### {lg}", "", "```", open(p, errors="replace").read()[:1500], "```", ""]
    for lg in sorted(glob.glob(f"{ST}/replay4_*.log") + glob.glob(f"{ST}/replay5_*.log")):
        pr = [l for l in open(lg, errors="replace") if l.startswith("PROBE")]
        if pr: L += [f"### {os.path.basename(lg)} (A2 pool probe)", "", "```", pr[-1][:1500], "```", ""]
    L += ["## 6. 실행 원장·오류", "", "```", open(f"{ST}/chain.log").read()[-6000:] if os.path.exists(f"{ST}/chain.log") else "", "```", ""]
    ev = [json.loads(l) for l in open(f"{ST}/execution_events.jsonl")] if os.path.exists(f"{ST}/execution_events.jsonl") else []
    L += [f"원장 event 수 {len(ev)}: " + json.dumps({k: sum(1 for e in ev if e['event'] == k) for k in sorted(set(e['event'] for e in ev))}, ensure_ascii=False), "", "## 7. 파일", "", "- 문서: " + ", ".join(sorted(os.path.basename(x) for x in glob.glob(f"{FEAT}/*.md") + glob.glob(f"{FEAT}/*.yaml") + glob.glob(f"{FEAT}/*.json"))), f"- 원자료: {os.path.relpath(CAMP, REPO)}/ (qwen/OPT4/<boot>/<session>/, replay/, calibration/, placements/, variants/, state/)", "- 도구: eval/ide076/ (TOOL_INTERFACES.md)"]
    open(f"{FEAT}/FULL_REPORT.md", "w").write("\n".join(L) + "\n")
    # ARTIFACT_INDEX / SHA256SUMS
    idx = []
    for p in sorted(glob.glob(f"{CAMP}/**/*", recursive=True)):
        if os.path.isfile(p) and os.path.getsize(p) < 200 * 1024 * 1024: idx.append((os.path.relpath(p, REPO), os.path.getsize(p), sha_plain(p)))
    with open(f"{FEAT}/ARTIFACT_INDEX.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(["path", "bytes", "sha256"]); w.writerows(idx)
    with open(f"{FEAT}/SHA256SUMS.txt", "w") as f:
        for p, b, h in idx: f.write(f"{h}  {p}\n")
    print("FULL_REPORT", len(L), "lines; artifacts", len(idx))


if __name__ == "__main__": main()
