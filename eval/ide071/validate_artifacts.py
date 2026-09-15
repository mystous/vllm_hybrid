#!/usr/bin/env python3
"""IDE_071 — 결과 문서 자동 검증 (compact §8 / 12단계 §18.8). 결과: shadow_assists/features/IDE_071/VALIDATION.md (통과/실패 항목 나열)."""
import glob, gzip, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
CAMP = f"{REPO}/eval/results/IDE_071_20260916"; CD = f"{CAMP}/compact"
checks = []
def chk(name, ok, detail=""): checks.append((name, bool(ok), detail))

def main():
    cells = [json.loads(l) for l in open(f"{FEAT}/manifests/cells.jsonl")]
    ids = {(c["cell_id"], c["attempt_id"]) for c in cells}
    # 1. 등록 셀 = 상태표 셀 (FULL_REPORT §3 표 행)
    fr = open(f"{FEAT}/FULL_REPORT.md").read()
    rows = set(re.findall(r"^\| ([A-Za-z0-9_]+) \| (?:P\d|COMPACT|None) \| [^|]* \| (a\d) \| ", fr, re.M))
    chk("registered_cells_eq_status_rows", ids == rows, f"cells.jsonl {len(ids)} vs 상태표 {len(rows)} diff={sorted(ids ^ rows)[:10]}")
    # 2. 완료 반복 수 = raw benchmark 파일 수; 요청 합계 = requests.jsonl 행 수
    bad = []
    for c in cells:
        d = f"{REPO}/{c['dir']}/"; stt = json.load(open(f"{d}status.json"))
        for r in stt.get("reps") or []:
            rd = f"{d}{r['rep_id']}/"
            if r["valid"]:
                if not os.path.exists(f"{rd}benchmark.raw.json"): bad.append((c["cell_id"], r["rep_id"], "no raw"))
                n = sum(1 for _ in open(f"{rd}requests.jsonl")) if os.path.exists(f"{rd}requests.jsonl") else -1
                if n != r["n"] or (r.get("completed") or 0) + (r.get("failed") or 0) != n: bad.append((c["cell_id"], r["rep_id"], f"rows {n} n {r['n']} completed {r.get('completed')} failed {r.get('failed')}"))
                try: json.load(open(f"{rd}benchmark.raw.json"))
                except Exception as e: bad.append((c["cell_id"], r["rep_id"], f"raw json parse {e}"))
    chk("valid_reps_have_raw_and_request_rows_match", not bad, str(bad[:10]))
    # 3. 성능값 raw 경로·계산식 존재
    chk("calc_rules_section", "## 6. 계산 규칙" in fr and "output_tps =" in fr)
    # 4. config/build/hotmap hash
    chk("config_hash_all", all(c.get("config_hash") for c in cells), [c["cell_id"] for c in cells if not c.get("config_hash")][:5])
    chk("code_sha_all", all(c.get("code_sha", {}).get("kt_kernel_ext_sha256") for c in cells))
    env = json.load(open(f"{FEAT}/evidence/software_manifest.json"))
    chk("hotmap_hashes", all(env["files"][k]["sha256"] and not env["files"][k]["sha256"].startswith("ERR") for k in ("hotmap_v1", "hotmap_v2", "layer_budget_5952", "hotmap_mixed_0.25")))
    # 5. 단위·측정창·precision·cache 정책
    chk("workload_table", "## 7. 워크로드" in fr and "cache_policy" in fr)
    chk("precision_family_all", all(c.get("precision_family") for c in cells))
    # 6. 링크 대상 존재
    links = re.findall(r"\]\(([^)]+\.md)\)", fr); missing = [l for l in links if not os.path.exists(f"{FEAT}/{l}")]
    chk("md_links_exist", not missing, missing[:5])
    # 7. 원본 SHA256 검증 (artifact_manifest 표본 200 + 전체 크기)
    man = [json.loads(l) for l in open(f"{FEAT}/manifests/artifact_manifest.jsonl")]; badsha = []
    import random; random.seed(1)
    for j in random.sample(man, min(200, len(man))):
        p = f"{REPO}/{j['path']}"
        if not os.path.exists(p) or os.path.getsize(p) != j["bytes"] or sha256(p) != j["sha256"]: badsha.append(j["path"])
    chk("artifact_sha256_sample200", not badsha, badsha[:5])
    gz = [f"{REPO}/{j['path']}" for j in man if j["path"].endswith("server.full.log.gz")]; badgz = []
    for p in gz:
        try:
            with gzip.open(p, "rb") as f: f.read(1 << 16)
        except Exception as e: badgz.append(p)
    chk("server_log_gzip_integrity", not badgz, badgz[:3])
    # 8. FAILED/UNSUPPORTED 값이 통계에 없음: stats 표의 셀은 valid 반복만 (render 는 valid 필터) — 실패 셀 ID 가 통계표에 없는지
    completed_ids = {c["cell_id"] for c in cells if c["status"] == "COMPLETED"}
    failed = [c["cell_id"] for c in cells if c["status"] != "COMPLETED" and c["cell_id"] not in completed_ids]   # 다른 attempt 가 COMPLETED 면 제외
    stats_txt = "\n".join(re.findall(r"### [^\n]*반복 통계[^\n]*\n(.*?)(?=\n##|\Z)", fr, re.S))
    leak = [f for f in failed if re.search(rf"^\| {re.escape(f)} \| ", stats_txt, re.M)]
    chk("failed_cells_not_in_stats", not leak, leak)
    # 9. 동일 이름 덮어쓰기 없음: attempt 디렉터리 유일
    dup = [k for k in ids if list(ids).count(k) > 1]; chk("no_duplicate_cell_attempt", not dup)
    # 10. 해석 문장 금지 키워드
    banned = ["권장", "최적", "병목은", "효과가 없", "채택", "best practice", "추천"]
    hits = [b for b in banned if b in fr.split("## 11.")[0]]
    chk("no_evaluation_keywords_in_report_body", not hits, hits)
    # 11. CSV 행 수 vs 측정 창 (cpu_timeseries 1 s 샘플 ≥ duration*0.8)
    short = []
    for c in cells:
        d = f"{REPO}/{c['dir']}/"; stt = json.load(open(f"{d}status.json"))
        for r in stt.get("reps") or []:
            if not r["valid"]: continue
            p = f"{d}{r['rep_id']}/cpu_timeseries.csv"
            if os.path.exists(p):
                n = sum(1 for _ in open(p)) - 1
                if n < 0.8 * (r.get("duration") or 0): short.append((c["cell_id"], r["rep_id"], n, r.get("duration")))
    chk("cpu_timeseries_covers_measurement", not short, str(short[:5]))
    ok = sum(1 for _, o, _ in checks if o)
    L = [f"# VALIDATION — IDE_071 ({now()['wall_kst']})", "", f"통과 {ok}/{len(checks)}", "", "| check | 결과 | 상세 |", "|---|---|---|"]
    for n, o, d in checks: L.append(f"| {n} | {'PASS' if o else 'FAIL'} | {str(d)[:300]} |")
    open(f"{FEAT}/VALIDATION.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
