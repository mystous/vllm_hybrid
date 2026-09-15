#!/usr/bin/env python3
"""IDE_071 — 30분 보고 루프 (지시서 §17). 60 s 마다 state 를 읽고, 시작 monotonic 기준 1,800 s 경계마다 progress/<ts>.md 저장 + PROGRESS.md 갱신 + delivery_log 에 saved 기록.
사용자 실시간 전달은 세션 cron (에이전트) 이 이 파일을 읽어 수행하고 delivered 로 기록한다. 이 프로세스 자체는 REPORT_DELIVERY_UNAVAILABLE 상태만 남긴다.
사용: report_progress.py <campaign_dir>
"""
import json, os, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def counts(st):
    c = {"registered": 0, "completed": 0, "failed": 0, "blocked": 0, "pending": 0, "other": 0}
    for cid, rec in st.get("cells", {}).items():
        c["registered"] += 1
        ex = [a["exit_status"] for a in rec.get("attempts", [])]
        if not ex: c["pending"] += 1
        elif "COMPLETED" in ex: c["completed"] += 1
        elif any(e.startswith("BLOCKED") or e in ("UNSUPPORTED", "NO_EFFECTIVE_PATH") for e in ex): c["blocked"] += 1
        elif any(e.startswith("FAILED") or e in ("TIMEOUT", "INVALID_CONFIG") for e in ex): c["failed"] += 1
        else: c["other"] += 1
    return c

def latest_meas(camp, k=6):
    rows = []
    for f in sorted(glob.glob(f"{camp}/*/a*/*/metrics.json"), key=os.path.getmtime)[-k:]:
        m = json.load(open(f)); parts = f.split("/")
        rows.append(f"{parts[-4]}/{parts[-3]} {m['rep_id']}: out_tps={m.get('output_tps')} ttft50={m.get('ttft_p50')} tpot50={m.get('tpot_p50')} ok={m.get('completed')}/{m.get('n')} valid={m.get('valid')}")
    return rows

def latest_error(camp):
    fs = sorted(glob.glob(f"{camp}/*/a*/errors.jsonl"), key=os.path.getmtime)
    if not fs: return "없음"
    ls = open(fs[-1]).read().strip().splitlines()
    if not ls: return "없음"
    e = json.loads(ls[-1]); return f"{e.get('t',{}).get('wall_kst','?')} {fs[-1].split('/')[-3]} stage={e.get('stage')} {str(e.get('reason') or e.get('msg') or e.get('exception') or e.get('verdict'))[:200]} log={fs[-1]}"

def write_report(camp, st, t0, feat):
    t = now(); ts = time.strftime("%Y%m%dT%H%M%S+0900", time.localtime(t["epoch"]))
    c = counts(st); cur = st.get("current") or {}
    cur_state = "IDLE"
    if cur.get("cell"):
        sp = f"{camp}/{cur['cell']}/{cur['attempt']}/status.json"
        if os.path.exists(sp): s = json.load(open(sp)); cur_state = s.get("state"); cur["rep"] = s.get("current_rep")
    hbm = sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\\n' ';'").stdout.strip()
    ram = sh("free -g | awk 'NR==2{print \"used \"$3\" GB / free \"$4\" GB\"}'").stdout.strip(); disk = sh("df -h /data | awk 'NR==2{print $4\" avail\"}'").stdout.strip()
    arts = sorted(glob.glob(f"{camp}/*/a*/*/metrics.json"), key=os.path.getmtime)[-3:]
    hist = st.get("history", [])[-6:]
    nxt = []
    try:
        ph = st.get("phase"); cells = [json.loads(l)["cell_id"] for l in open(f"{camp}/manifests/cells_{ph}.jsonl")]
        nxt = [x for x in cells if x not in st.get("cells", {})][:5]
    except Exception: pass
    comp = ""
    cp = f"{camp}/compact/compact_state.json"
    if os.path.exists(cp):
        cs = json.load(open(cp)); b = cs.get("budget", {}); cc = cs.get("current") or {}
        done = {k: v["attempts"][-1]["exit_status"] for k, v in cs.get("cells", {}).items()}
        last = []
        for f in sorted(glob.glob(f"{camp}/compact/*/a*/*/metrics.json"), key=os.path.getmtime)[-4:]:
            m = json.load(open(f)); last.append(f"{f.split('/')[-4]} {m['rep_id']}: out_tps={m.get('output_tps')} ttft95={m.get('ttft_p95')} tpot95={m.get('tpot_p95')} ok={m.get('completed')}/{m.get('n')} valid={m.get('valid')}")
        comp = f"""
## compact (cpu_offload_no_02_compact)
- compact_status: {cs.get('status')} · current: {cc.get('cell')}/{cc.get('attempt')}
- 일반 벤치 소비 {b.get('bench')}/{b.get('bench_max')} · 진단 {b.get('diag')} · 연속부하 {b.get('load')} · 재시도 {b.get('retry')} · GSM {b.get('gsm')}
- 부팅 소비 {b.get('boot')}/{b.get('boot_max')} · 셀 상태: {done}
- 재사용 대조군: {', '.join(f"{k}={v['source_cell']} mean {v['mean']:.1f}" for k, v in cs.get('reused', {}).items())}
- 직전 실측: """ + " | ".join(last) + f"""
- targets: {cs.get('targets')}
"""
    md = comp + f"""# 중간 실행 보고 — {t['wall_kst']}

- campaign_id: {st.get('campaign_id')}
- elapsed_seconds: {int(t['monotonic'] - t0)}
- utc: {t['wall_utc']}
- current_phase / cell_id / attempt / rep_id: {st.get('phase')} / {cur.get('cell')} / {cur.get('attempt')} / {cur.get('rep')}
- phase_state: {cur_state}
- registered / completed / failed / blocked / pending: {c['registered']} / {c['completed']} / {c['failed']} / {c['blocked']} / {c['pending']}
- last_completed_cells: {', '.join(f"{h['cell']}={h['status']}" for h in hist) or '없음'}
- latest_measurements:
""" + "\n".join(f"  - {r}" for r in latest_meas(camp)) + f"""
- latest_error: {latest_error(camp)}
- retries_and_recovery: {sum(1 for r in st.get('cells',{}).values() if len(r.get('attempts',[]))>1)} 셀 재시도
- HBM: {hbm}
- RAM: {ram}; disk /data: {disk}
- latest_persisted_artifacts: {', '.join(arts) or '없음'}
- next_registered_cells: {', '.join(nxt) or '없음'}
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)
"""
    os.makedirs(f"{feat}/progress", exist_ok=True)
    p = f"{feat}/progress/{ts}.md"; open(p, "w").write(md)
    with open(f"{feat}/PROGRESS.md", "a") as f: f.write("\n" + md)
    jappend({"t": t, "file": p, "state": "saved", "channel": "file"}, f"{feat}/progress/delivery_log.jsonl")
    return p

def main():
    camp = sys.argv[1]; feat = FEAT
    t0 = time.monotonic(); k = 0
    mono_file = f"{camp}/state/report_loop.json"; jdump({"t_start": now(), "interval_s": 1800}, mono_file)
    write_report(camp, json.load(open(f"{camp}/state/state.json")) if os.path.exists(f"{camp}/state/state.json") else {"campaign_id": os.path.basename(camp)}, t0, feat)
    while True:
        time.sleep(30)
        el = time.monotonic() - t0
        if el >= (k + 1) * 1800:
            k += 1
            try:
                st = json.load(open(f"{camp}/state/state.json")) if os.path.exists(f"{camp}/state/state.json") else {"campaign_id": os.path.basename(camp)}
                write_report(camp, st, t0, feat)
            except Exception as e:
                jappend({"t": now(), "state": "failed", "error": str(e)}, f"{feat}/progress/delivery_log.jsonl")
        if os.path.exists(f"{camp}/state/CAMPAIGN_DONE"):
            st = json.load(open(f"{camp}/state/state.json")); write_report(camp, st, t0, feat); break

if __name__ == "__main__":
    main()
