#!/usr/bin/env python3
"""IDE_072 — 30분 보고 루프 (§5.3). 1,800 s 경계마다 progress/<ts>.md + PROGRESS.md 갱신, delivery_log 에 saved. 사용자 전달은 세션 cron 이 수행."""
import json, os, sys, time, glob
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_072")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
CAMP = f"{REPO}/eval/results/IDE_072_20260916"

def report(t0):
    t = now(); ts = time.strftime("%Y%m%dT%H%M%S+0900", time.localtime(t["epoch"]))
    st = json.load(open(f"{CAMP}/state/state.json")) if os.path.exists(f"{CAMP}/state/state.json") else {}
    u = st.get("usage", {}); b = st.get("budget", {}); cur = st.get("current") or {}
    stage = "IDLE"
    if cur.get("cell"):
        p = f"{CAMP}/{cur['cell']}/{cur['attempt']}/status.json"
        if os.path.exists(p): stage = json.load(open(p)).get("state")
    done = {k: v["attempts"][-1]["exit_status"] for k, v in st.get("cells", {}).items()}
    last = []
    for f in sorted(glob.glob(f"{CAMP}/*/a*/*/metrics.json"), key=os.path.getmtime)[-4:]:
        m = json.load(open(f)); last.append(f"{f.split('/')[-4]} {m.get('rep_id') or 'REPLAY'}: out_tps={m.get('output_tps')} ok={m.get('completed')}/{m.get('n', 32)} valid={m.get('valid', m.get('healthy_after'))}")
    errs = sorted(glob.glob(f"{CAMP}/*/a*/errors.jsonl"), key=os.path.getmtime)
    le = "없음"
    if errs:
        ls = open(errs[-1]).read().strip().splitlines()
        if ls: e = json.loads(ls[-1]); le = f"{errs[-1].split('/')[-3]} stage={e.get('stage')} {str(e.get('reason') or e.get('verdict') or e.get('msg'))[:200]} ({errs[-1]})"
    nsm = len(glob.glob(f"{CAMP}/*/a*/smoke_greedy4.json")); nwm = len(glob.glob(f"{CAMP}/*/a*/warmup/bench_summary.txt"))
    md = f"""# 중간 실행 보고 — {t['wall_kst']} (IDE_072)

- 경과: {int(t['monotonic'] - t0)} s · 현재 cell·stage·run_id: {cur.get('cell')} / {stage} / {cur.get('cell')}.{cur.get('attempt')}
- PERF {u.get('PERF')}/{b.get('PERF')} · REPLAY {u.get('REPLAY')}/{b.get('REPLAY')} · LOAD {u.get('LOAD')}/{b.get('LOAD')} · DIAG {u.get('DIAG')}/{b.get('DIAG')} · RETRY {u.get('RETRY')}/{b.get('RETRY')}
- 부하 세션 {u.get('SESSIONS')}/{b.get('SESSIONS')} · 부팅 {u.get('BOOT')}/{b.get('BOOT')} · 워밍업 {nwm} · smoke {nsm} · GSM 문항 {u.get('GSM_Q')}/{b.get('GSM_Q')}
- 셀 상태: {done}
- 직전 원측정값: {' | '.join(last) or '없음'}
- 오류: {le}
- 다음 등록 항목: {[x for x in ('V0_first','V1_first','V2_first','Q0_first','V0_confirm','V1_confirm','LOAD','D0(조건부)') if x not in done][:4]}
- 저장된 최신 MD: {sorted(glob.glob(FEAT + '/progress/*.md'))[-1] if glob.glob(FEAT + '/progress/*.md') else '없음'}
- 전달: saved (세션 cron 이 사용자 전달 후 delivered 기록)
"""
    os.makedirs(f"{FEAT}/progress", exist_ok=True); p = f"{FEAT}/progress/{ts}.md"; open(p, "w").write(md)
    with open(f"{FEAT}/PROGRESS.md", "a") as f: f.write("\n" + md)
    jappend({"t": t, "file": p, "state": "saved"}, f"{FEAT}/progress/delivery_log.jsonl")

def main():
    t0 = time.monotonic(); k = 0; report(t0)
    while True:
        time.sleep(30)
        if time.monotonic() - t0 >= (k + 1) * 1800: k += 1; report(t0)
        if os.path.exists(f"{CAMP}/state/CAMPAIGN_DONE"): report(t0); break

if __name__ == "__main__":
    main()
