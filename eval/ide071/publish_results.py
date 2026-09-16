#!/usr/bin/env python3
"""IDE_071 — 게시 (compact §8 / 12단계 §19.3). 경로 명시 stage → data commit → push → 원격 SHA·파일 확인 → PUBLISH_RECEIPT.md → receipt commit → push.
force push 금지. 5 MB 초과·.pt·perf.data 제외. 사용: publish_results.py"""
import glob, json, os, subprocess, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
CAMP = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/eval/results/IDE_071_20260916"   # 캠페인 결과 루트 (IDE_072 는 인자로)
BR = "feat/cpu-offload-ide071"

def git(cmd, check=True):
    r = subprocess.run(f"git -C {REPO} {cmd}", shell=True, capture_output=True, text=True)
    if check and r.returncode != 0: raise RuntimeError(f"git {cmd}: {r.stderr[-800:]}")
    return r.stdout.strip()

def main():
    rec = {"t": now(), "branch": git("branch --show-current"), "remote": re.sub(r"https://[^@]*@", "https://", git("remote get-url origin"))}
    assert rec["branch"] == BR, rec["branch"]
    # 게시 대상: feature 디렉터리 전체 + 하네스 + 결과 (제외 규칙)
    paths = [f"{FEAT}", f"{REPO}/eval/ide071", f"{REPO}/shadow_assists/id_registry.md", f"{REPO}/shadow_assists/README.md"]
    excluded = []
    files = []
    for p in sorted(glob.glob(f"{CAMP}/**/*", recursive=True)):
        if os.path.isdir(p): continue
        if os.path.getsize(p) > 5 * 1024 * 1024 or p.endswith(".pt") or p.endswith("perf.data"): excluded.append((os.path.relpath(p, REPO), os.path.getsize(p))); continue
        files.append(p)
    # 비밀정보 검사 (토큰 패턴)
    leak = []
    for p in files + [x for x in glob.glob(f"{FEAT}/**/*", recursive=True) if os.path.isfile(x)]:
        if p.endswith((".gz", ".json", ".csv", ".jsonl", ".md", ".txt", ".sh", ".log", ".py")):
            try: s = open(p, "rb").read(2_000_000)
            except Exception: continue
            if re.search(rb"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})", s): leak.append(os.path.relpath(p, REPO))
    if leak: raise SystemExit(f"secret pattern found, abort: {leak[:5]}")
    git("add -- " + " ".join(f'"{x}"' for x in paths))
    for i in range(0, len(files), 400): git("add -- " + " ".join(f'"{x}"' for x in files[i:i + 400]))
    staged = git("diff --cached --name-only").splitlines()
    big = [l for l in git("diff --cached --numstat").splitlines() if l.split()[0] != "-" and int(l.split()[0]) > 200000]
    rec["staged_files"] = len(staged); rec["excluded_large"] = excluded[:50]; rec["excluded_count"] = len(excluded)
    TAG = os.path.basename(FEAT)
    msg = f"{TAG} data: 전 셀 원자료·FULL_REPORT·RESULT·manifest ({os.path.basename(CAMP)})\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01DLNvsR6jaGRWLwGy64zHCh"
    if staged:
        git(f"commit -q -m {json.dumps(msg)}")
    rec["data_commit"] = git("rev-parse HEAD")
    push = subprocess.run(f"git -C {REPO} push origin {BR}", shell=True, capture_output=True, text=True)
    rec["push_rc"] = push.returncode; rec["push_stderr"] = push.stderr[-500:]
    remote_sha = git(f"ls-remote origin refs/heads/{BR}", check=False).split()[0] if push.returncode == 0 else None
    rec["remote_sha"] = remote_sha; rec["remote_matches"] = (remote_sha == rec["data_commit"])
    # 원격 파일 존재 확인 (gh 또는 raw)
    exists = {}
    subprocess.run(f"git -C {REPO} fetch -q origin {BR}", shell=True)
    for f in (os.path.relpath(FEAT, REPO) + "/FULL_REPORT.md", os.path.relpath(FEAT, REPO) + "/RESULT.md"):
        r = subprocess.run(f"git -C {REPO} cat-file -s origin/{BR}:{f}", shell=True, capture_output=True, text=True)
        exists[f] = {"rc": r.returncode, "size": r.stdout.strip()[:20], "err": r.stderr.strip()[:200], "method": "git fetch + cat-file -s origin/branch:path (gh 미인증)"}
    rec["remote_files"] = exists
    status = "PUBLISHED" if push.returncode == 0 and rec["remote_matches"] and all(v["rc"] == 0 for v in exists.values()) else ("PUBLISH_FAILED" if push.returncode != 0 else "PUSHED_UNVERIFIED")
    rec["status"] = status
    L = [f"# PUBLISH_RECEIPT — IDE_071 ({rec['t']['wall_kst']})", "", f"- 상태: **{status}**", f"- remote: {rec['remote']} · branch: {BR}", f"- data commit: `{rec['data_commit']}` · remote SHA: `{remote_sha}` · 일치: {rec['remote_matches']}",
         f"- push rc {push.returncode} {push.stderr.strip()[-200:]}", f"- staged 파일 {len(staged)} · 제외(5 MB 초과/.pt/perf.data) {len(excluded)} 개 (서버 보존, manifests/artifact_manifest.jsonl 참조)",
         "- 원격 파일 확인: " + "; ".join(f"{k}: rc={v['rc']} size={v['size']}" for k, v in exists.items()),
         f"- GitHub 위치: https://github.com/mystous/vllm_hybrid/blob/{rec['data_commit']}/{os.path.relpath(FEAT, REPO)}/FULL_REPORT.md , https://github.com/mystous/vllm_hybrid/blob/{rec['data_commit']}/{os.path.relpath(FEAT, REPO)}/RESULT.md",
         f"- raw 다운로드: https://raw.githubusercontent.com/mystous/vllm_hybrid/{rec['data_commit']}/{os.path.relpath(FEAT, REPO)}/FULL_REPORT.md", "", "## 제외 대용량 파일 (상위 50)", "", "| path | bytes |", "|---|---|"] + [f"| {p} | {b} |" for p, b in excluded[:50]]
    open(f"{FEAT}/PUBLISH_RECEIPT.md", "w").write("\n".join(L) + "\n"); jdump(rec, f"{FEAT}/manifests/publish_receipt.json")
    git(f"add -- {FEAT}/PUBLISH_RECEIPT.md {FEAT}/manifests/publish_receipt.json")
    git(f'commit -q -m "{TAG}: PUBLISH_RECEIPT (receipt commit)\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01DLNvsR6jaGRWLwGy64zHCh"')
    p2 = subprocess.run(f"git -C {REPO} push origin {BR}", shell=True, capture_output=True, text=True)
    rec["receipt_commit"] = git("rev-parse HEAD"); rec["receipt_push_rc"] = p2.returncode; jdump(rec, f"{FEAT}/manifests/publish_receipt.json")
    print(json.dumps({k: rec[k] for k in ("status", "data_commit", "remote_sha", "remote_matches", "receipt_commit", "receipt_push_rc", "staged_files", "excluded_count")}, indent=1))

if __name__ == "__main__":
    main()
