#!/usr/bin/env python3
"""IDE_075 A01 — IDE_074 세션의 시간창 복원 + PCM v2 재집계 + perf 누적값 처리 (서버 실행 없음).
창(§4.3): client_process_window(bench 호출~종료, float epoch→ns, precision 1 µs), client_request_window(ESTIMATED: 요청별 timestamp 없음 → t_end−benchmark_duration 역산은 같은 기준 증거 없음 → ESTIMATED_WINDOW),
server_service_window(서버 로그 초 해상도 → 범위), forward_envelope(P1: 트레이스 step annotation 첫 시작~마지막 종료 / P1·P2: kt_evt 첫 t_go ~ 마지막 max(t_done_set,t_def_end)), phase_intervals(트레이스 step[...] 구간), collector_window(actual_start/end, PCM 은 표본 구간 union), drain_window.
PCM: 소수초·시간대(Asia/Seoul, 세션 metadata: 수집 호스트 로컬 시각) 보존, 실패 표본 INVALID_TIMESTAMP, 표본 구간 = [ts−delay, ts) (interval_contract=ASSUMED_END_TIMESTAMP, 근거: 첫 표본 시각 − collector actual_start ≈ delay), 완전 포함 표본 시간 가중 평균 + 경계 비례 추정 분리.
perf: 누적값 하나뿐 → 비례 환산 금지, legacy 값만 보존 (NOT_RECOVERABLE).
산출: eval/results/IDE_075_20260917/offline/{legacy_windows.csv, pcm_samples_v2.csv, pcm_window_summary_v2.csv, timestamp_parse_errors.jsonl}, features/IDE_075/WINDOW_RECONSTRUCTION.md"""
import json, os, sys, csv, gzip, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsparse import local_timestamp_ns, iso_timestamp_ns, epoch_float_to_ns, classify_samples
HOME = os.path.expanduser("~"); REPO = f"{HOME}/projects/vllm_hybrid"; C74 = f"{REPO}/eval/results/IDE_074_20260917"; KT_HOST = f"{HOME}/.cache/huggingface/kt/ide074"
OUT = f"{REPO}/eval/results/IDE_075_20260917/offline"; FEAT = f"{REPO}/shadow_assists/features/IDE_075"; os.makedirs(OUT, exist_ok=True)
PCM_TZ = "Asia/Seoul"   # pcm-memory 는 수집 호스트(violet-h100-016, TZ=KST) 로컬 시각을 출력 (metrics.json 의 wall_kst 와 같은 호스트)
PCM_DELAY_NS = 1_000_000_000


def sessions():
    for bd in sorted(glob.glob(f"{C74}/*/*/*/")):
        for sd in sorted(glob.glob(f"{bd}*/")):
            if os.path.exists(f"{sd}metrics.json") and not sd.rstrip("/").endswith("warmup"): yield sd, json.load(open(f"{sd}metrics.json"))


def kt_evt_envelope(boot, t0_ns, t1_ns):
    p = f"{KT_HOST}/{boot}/kt_evt.csv"
    if not os.path.exists(p): return None
    first = None; last = 0; n = 0
    for r in csv.DictReader(l for l in open(p) if not l.startswith("#")):
        g = int(r["t_go"])
        if g < t0_ns or g > t1_ns: continue
        n += 1; first = g if first is None else min(first, g); last = max(last, int(r["t_done_set"]), int(r["t_def_end"]), g)
    return (first, last, n) if n else None


def trace_envelope(boot, trace_files):
    tf = next((f"{KT_HOST}/{boot}/profiles/{n}" for n, _ in trace_files if "TP-0" in n), None)
    if not tf or not os.path.exists(tf): return None, []
    j = json.load(gzip.open(tf)); base = j.get("baseTimeNanoseconds", 0)
    steps = [(base + int(round(e["ts"] * 1000)), base + int(round((e["ts"] + e["dur"]) * 1000)), e["name"]) for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[")]
    steps.sort(); return ((steps[0][0], steps[-1][1]) if steps else None), steps


def main():
    W = []; PS = []; PSUM = []; ERR = open(f"{OUT}/timestamp_parse_errors.jsonl", "w"); MD = ["# WINDOW_RECONSTRUCTION — IDE_075 (IDE_074 세션 재구성)", "", "창 정의는 스크립트 docstring. 단위 ns 정수. `ESTIMATED_WINDOW` 는 같은 기준 증거가 없는 역산.", ""]
    for sd, m in sessions():
        sid = m["session"]; boot = m["boot_id"]; t0 = epoch_float_to_ns(m["t_start"]["epoch"]); t1 = epoch_float_to_ns(m["t_end"]["epoch"]); dr = epoch_float_to_ns(m["drain"]["t_drain_end"]["epoch"])
        row = {"session": sid, "boot": boot, "mode": m["mode"], "client_process_start_ns": t0, "client_process_end_ns": t1, "client_process_precision": "float epoch (~1 µs)", "benchmark_duration_s": (m.get("summary") or {}).get("duration"),
               "client_request_window": "ESTIMATED_WINDOW (요청별 timestamp 없음; t_end−duration 역산의 기준 동일성 미증명)", "server_service_first_receive_utc_s": (m.get("first_request_server_log") or {}).get("wall_utc_s_res"), "server_service_precision": "1 s (로그 초 해상도)",
               "drain_end_ns": dr, "drain_seconds": m["drain"]["drain_seconds"]}
        prof = next((c for c in m["observer"]["collectors"] if c["collector_id"] == "torch_profiler"), None)
        env_tr, steps = trace_envelope(boot, prof["trace_files"]) if prof else (None, [])
        env_evt = kt_evt_envelope(boot, t0, dr)
        row.update({"forward_envelope_trace_start_ns": env_tr[0] if env_tr else None, "forward_envelope_trace_end_ns": env_tr[1] if env_tr else None, "phase_intervals_n": len(steps), "phase_names": json.dumps(dict(collections.Counter(s[2][:24] for s in steps)), ensure_ascii=False) if steps else None,
                    "forward_envelope_evt_start_ns": env_evt[0] if env_evt else None, "forward_envelope_evt_end_ns": env_evt[1] if env_evt else None, "kt_evt_records": env_evt[2] if env_evt else None,
                    "envelope_source": "trace step annotation" if env_tr else ("kt_evt (첫 go ~ 마지막 done/def_end)" if env_evt else "NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음)"),
                    "lead_client_to_forward_s": ((env_tr or env_evt)[0] - t0) / 1e9 if (env_tr or env_evt) else None, "tail_forward_end_to_client_end_s": (t1 - (env_tr or env_evt)[1]) / 1e9 if (env_tr or env_evt) else None})
        for c in m["observer"]["collectors"]:
            cs = (c.get("actual_start") or c.get("arm_time") or {}).get("epoch"); ce = (c.get("actual_end") or {}).get("epoch")
            row[f"collector_{c['collector_id'].split('(')[0]}_start_ns"] = epoch_float_to_ns(cs) if cs else None; row[f"collector_{c['collector_id'].split('(')[0]}_end_ns"] = epoch_float_to_ns(ce) if ce else None
        # PCM v2
        pcm = f"{sd}pcm_memory.csv"
        if os.path.exists(pcm):
            rows = list(csv.reader(open(pcm, errors="replace"))); hdr = [f"{a.strip()}|{b.strip()}" for a, b in zip(rows[0], rows[1])]
            ix = {k: [i for i, h in enumerate(hdr) if h == k] for k in ("System|Read", "System|Write", "System|Memory", "SKT0|Memory", "SKT1|Memory")}
            di = hdr.index("|Date") if "|Date" in hdr else 0; ti = hdr.index("|Time") if "|Time" in hdr else 1
            samples = {k: [] for k in ix}; n_bad = 0; prev_ts = None; gaps = 0
            for r in rows[2:]:
                if len(r) < len(hdr) - 1: n_bad += 1; ERR.write(json.dumps({"session": sid, "reason": "SHORT_ROW", "row": r[:3]}) + "\n"); continue
                try: ts = local_timestamp_ns(r[di], r[ti], PCM_TZ)
                except ValueError as e: n_bad += 1; ERR.write(json.dumps({"session": sid, "reason": f"INVALID_TIMESTAMP:{e}", "row": r[:2]}) + "\n"); continue
                if prev_ts is not None and ts - prev_ts > PCM_DELAY_NS * 1.5: gaps += 1
                prev_ts = ts; iv = (ts - PCM_DELAY_NS, ts)
                for k, idx in ix.items():
                    if idx:
                        try: v = float(r[idx[-1]])
                        except ValueError: continue
                        samples[k].append((iv, v)); PS.append({"session": sid, "metric": k, "sample_start_ns": iv[0], "sample_end_ns": iv[1], "value_MBps": v, "interval_contract": "ASSUMED_END_TIMESTAMP(delay 1 s)"})
            # 창: forward_envelope (evt 우선), legacy = client_process
            for wname, Wn in (("forward_envelope_evt", (env_evt[0], env_evt[1]) if env_evt else None), ("client_process_window(legacy)", (t0, t1))):
                if not Wn: PSUM.append({"session": sid, "window": wname, "status": "NOT_COLLECTED"}); continue
                for k in ix:
                    if not samples[k]: continue
                    cl = classify_samples(samples[k], Wn)
                    PSUM.append({"session": sid, "window": wname, "window_start_ns": Wn[0], "window_end_ns": Wn[1], "metric": k, **{kk: cl[kk] for kk in ("n_full", "n_boundary", "n_outside", "n_duplicate", "full_sample_coverage", "mean_full", "mean_overlap_est", "overlap_coverage")}, "invalid_rows": n_bad, "sample_gaps": gaps, "unit": "MB/s (pcm-memory; System 열 = host 합계, 소켓 합계와 재합산 금지)"})
        # perf: 누적값 → NOT_RECOVERABLE
        row["perf_stat_window"] = "NOT_RECOVERABLE (누적값 1개; 비례 환산 금지)" if os.path.exists(f"{sd}perf_stat.txt") else None
        W.append(row)
        MD.append(f"- {sid} ({m['mode']}, {boot}): client_process {(t1 - t0) / 1e9:.3f} s, envelope {row['envelope_source']}: lead {row['lead_client_to_forward_s'] and round(row['lead_client_to_forward_s'], 3)} s / tail {row['tail_forward_end_to_client_end_s'] and round(row['tail_forward_end_to_client_end_s'], 3)} s, phases {row['phase_names']}, drain {m['drain']['drain_seconds']:.1f} s")
    keys = []
    for r in W:
        for k in r:
            if k not in keys: keys.append(k)
    with open(f"{OUT}/legacy_windows.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(W)
    with open(f"{OUT}/pcm_samples_v2.csv", "w", newline="") as f:
        if PS: w = csv.DictWriter(f, fieldnames=list(PS[0].keys())); w.writeheader(); w.writerows(PS)
    keys = []
    for r in PSUM:
        for k in r:
            if k not in keys: keys.append(k)
    with open(f"{OUT}/pcm_window_summary_v2.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(PSUM)
    ERR.close()
    MD += ["", "## PCM 정정 (System|Read / Write, MB/s)", "", "| session | window | n_full/boundary/outside | full coverage | mean_full | mean_overlap_est | legacy(collector 전체 평균) |", "|---|---|---|---|---|---|---|"]
    leg = {}
    try:
        sys.path.insert(0, f"{REPO}/eval/ide074"); import render_report as R
        for sd, m in sessions():
            if os.path.exists(f"{sd}pcm_memory.csv"): leg[m["session"]] = R.parse_pcm(f"{sd}pcm_memory.csv", m["t_start"]["epoch"], m["t_end"]["epoch"])
    except Exception as e: MD.append(f"legacy parser 실패: {e}")
    for r in PSUM:
        if r.get("metric") in ("System|Read", "System|Write"):
            lg = leg.get(r["session"]) or {}; lv = lg.get("system_read_mean" if r["metric"] == "System|Read" else "system_write_mean")
            MD.append(f"| {r['session']} | {r['window']} | {r['metric']} {r['n_full']}/{r['n_boundary']}/{r['n_outside']} | {r['full_sample_coverage'] and round(r['full_sample_coverage'], 3)} | {r['mean_full'] and round(r['mean_full'], 1)} | {r['mean_overlap_est'] and round(r['mean_overlap_est'], 1)} | {lv and round(lv, 1)} |")
    MD += ["", "legacy 값은 `legacy_collector_window_mean` (bench 호출 창, 초 단위 파싱 실패 → 전 표본 포함). perf stat 은 세션 전체 누적값만 있어 부하 창 delta 복원 불가 (NOT_RECOVERABLE)."]
    open(f"{FEAT}/WINDOW_RECONSTRUCTION.md", "w").write("\n".join(MD) + "\n"); print("windows", len(W), "pcm samples", len(PS), "summaries", len(PSUM))


if __name__ == "__main__": main()
