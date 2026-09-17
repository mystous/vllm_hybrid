#!/usr/bin/env python3
"""IDE_073 — 30분 보고 루프 (§10.2). T0 = 준비 시작. monotonic 1,800 s 경계마다 progress/progress.jsonl + PROGRESS.md append. 상태는 state/RUN_STATE.json 요약만 읽음."""
import json, os, sys, time, glob
FEAT = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_073"); CAMP = os.path.expanduser("~/projects/vllm_hybrid/eval/results/IDE_073_20260917")
T0_WALL = float(sys.argv[1]) if len(sys.argv) > 1 else time.time()
def kst(t): return time.strftime("%Y-%m-%dT%H:%M:%S+0900", time.localtime(t))
def report(k):
    t = time.time(); st = {}
    p = f"{CAMP}/state/RUN_STATE.json"
    if os.path.exists(p):
        try: st = json.load(open(p))
        except Exception: st = {"_error": "RUN_STATE parse"}
    hbm = os.popen("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\\n' ' '").read().strip()
    rss = os.popen("ps -eo rss,comm --sort=-rss | awk 'NR>1 && NR<4{printf \"%s:%dMB \", $2, $1/1024}'").read().strip()
    disk = os.popen("df -h /data | awk 'NR==2{print $4}'").read().strip()
    dl = os.popen("du -sb /data/mystous/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8 2>/dev/null | cut -f1").read().strip()
    rec = {"scheduled_kst": kst(T0_WALL + k * 1800), "actual_kst": kst(t), "elapsed_s": int(t - T0_WALL), "phase": st.get("phase"), "current": st.get("current"), "counts": st.get("counts"),
           "done_last_30min": st.get("recent"), "glm_download_bytes": int(dl) if dl else None, "hbm_MiB": hbm, "top_rss": rss, "disk_avail": disk, "errors": st.get("errors_recent"), "next": st.get("next"),
           "eta": st.get("eta"), "delivery": "saved"}
    os.makedirs(f"{FEAT}/progress", exist_ok=True)
    with open(f"{FEAT}/progress/progress.jsonl", "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{FEAT}/PROGRESS.md", "a") as f:
        f.write(f"\n## 보고 {k} — 예정 {rec['scheduled_kst']} / 실제 {rec['actual_kst']} (경과 {rec['elapsed_s']} s)\n- 단계/현재: {rec['phase']} / {rec['current']}\n- 항목 수: {rec['counts']}\n- 최근 30분 완료: {rec['done_last_30min']}\n- GLM 다운로드 bytes: {rec['glm_download_bytes']}\n- HBM MiB: {hbm}\n- top RSS: {rss} · 디스크 여유: {disk}\n- 오류: {rec['errors']}\n- 다음: {rec['next']} · 잔여 추정: {rec['eta']}\n- 전달: saved (세션 cron 이 전달)\n")
k = 0; report(0)
while True:
    time.sleep(20)
    if time.time() >= T0_WALL + (k + 1) * 1800: k += 1; report(k)
    if os.path.exists(f"{CAMP}/state/CAMPAIGN_DONE"): report(k + 1); break
