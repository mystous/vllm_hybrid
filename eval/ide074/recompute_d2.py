#!/usr/bin/env python3
"""IDE_074 M1 — IDE_073 유효 D2 트레이스(bootA_retry2, TP0~3) 재집계.
장치별 (a) 전체 trace 창, (b) 실제 요청 창(bench t_start~t_end, host epoch) 의 §4.2 집계를 저장하고
원 보고값(IDE_073 layer_stage_aggregate)과 정정값·정정 사유를 나란히 기록한다.
MoE 경계는 이름 정규식(fused_moe_kernel) 휴리스틱이며 validity=HEURISTIC_MOE_BOUNDARY 로 표시한다.
기존 자료에 없는 layer/step/done 이벤트는 만들지 않는다."""
import gzip, json, os, sys, csv, re, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_union import device_window_metrics, union, clip, length, inter_boundary_gaps
REPO = os.path.expanduser("~/projects/vllm_hybrid"); FEAT = f"{REPO}/shadow_assists/features/IDE_074"
PROF = os.path.expanduser("~/.cache/huggingface/kt/ide073/profiles/qwen")
D2 = f"{REPO}/eval/results/IDE_073_20260917/qwen-PROBES/bootA_retry2/D2/metrics.json"
OLD = f"{REPO}/shadow_assists/features/IDE_073/profiles/qwen/layer_stage_aggregate.md"
OPS = ("kernel", "gpu_memcpy", "gpu_memset")


def load(tf):
    j = json.load(gzip.open(tf)); base_us = j.get("baseTimeNanoseconds", 0) / 1e3
    k, c, m, ann, moe = [], [], [], [], []
    for e in j["traceEvents"]:
        if e.get("ph") != "X": continue
        cat = e.get("cat"); s = base_us + e["ts"]; d = e.get("dur", 0) or 0
        if cat == "kernel":
            k.append((s, s + d))
            if re.search(r"fused_moe_kernel", e.get("name", "")): moe.append((s, s + d))
        elif cat == "gpu_memcpy": c.append((s, s + d))
        elif cat == "gpu_memset": m.append((s, s + d))
        elif cat == "gpu_user_annotation": ann.append((s, s + d))
    return {"trace_id": j.get("trace_id"), "base_ns": j.get("baseTimeNanoseconds"), "k": k, "c": c, "m": m, "ann": ann, "moe": sorted(moe), "n_events": len(j["traceEvents"])}


def old_values():
    """IDE_073 원 보고값 (layer_stage_aggregate.md GPU 타임라인 표) 파싱."""
    out = {}
    if not os.path.exists(OLD): return out
    for l in open(OLD):
        if l.startswith("| D2_") and "trace.json.gz" in l:
            c = [x.strip() for x in l.strip("|\n").split("|")]
            out[c[0]] = {"orig_gpu_ops": c[2], "orig_gpu_busy_s": c[3], "orig_span_s": c[4], "orig_busy_fraction": c[5], "orig_gap_field": c[6]}
    return out


def main():
    d2 = json.load(open(D2)); t0 = d2["t_start"]["epoch"] * 1e6; t1 = d2["t_end"]["epoch"] * 1e6
    old = old_values(); rows = []; gaps_all = {}
    os.makedirs(f"{FEAT}/profiles", exist_ok=True)
    for name, size in d2["trace_files"]:
        tf = f"{PROF}/{name}"; rank = int(re.search(r"TP-(\d+)", name).group(1)); T = load(tf)
        allops = T["k"] + T["c"] + T["m"]
        full_W = (min(a for a, b in allops), max(b for a, b in allops))
        for wname, W in (("trace_full", full_W), ("request_window", (t0, t1))):
            r = device_window_metrics(T["k"], T["c"], T["m"], W, annotations=T["ann"])
            mo = clip(T["moe"], W); g = inter_boundary_gaps(mo, allops) if len(mo) > 1 else []
            idle = [x["idle_in_gap_us"] for x in g]
            row = {"trace": name, "tp_rank": rank, "window": wname, "clock_domain": "host CLOCK_REALTIME (kineto baseTimeNanoseconds + ts); bench t_start/t_end = client host epoch(같은 호스트, 다른 컨테이너), 정렬 오차 미검증",
                   **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                   "request_coverage_fraction": round((min(full_W[1], t1) - max(full_W[0], t0)) / (t1 - t0), 4),
                   "moe_kernel_count_in_window": len(mo), "moe_boundary_method": "name regex fused_moe_kernel (HEURISTIC)", "inter_moe_gap_count": len(g),
                   "inter_moe_idle_p50_us": round(st.median(idle), 3) if idle else None, "inter_moe_idle_p95_us": round(sorted(idle)[int(round(.95 * (len(idle) - 1)))], 3) if idle else None,
                   "inter_moe_idle_max_us": round(max(idle), 3) if idle else None, "inter_moe_idle_sum_us": round(sum(idle), 3) if idle else None,
                   "validity": "CORRECTED_AGGREGATION;HEURISTIC_MOE_BOUNDARY;UNMEASURED_CLOCK_ALIGNMENT(client↔profiler)" if wname == "request_window" else "CORRECTED_AGGREGATION;HEURISTIC_MOE_BOUNDARY",
                   **old.get(name, {}), "correction_reason": "원 집계는 gpu_user_annotation 포함·stream 합계 혼재·요청 전 준비구간 포함; 정정은 kernel/memcpy/memset 만 clip 후 union"}
            rows.append(row); gaps_all[f"{name}:{wname}"] = g
        print(name, "events", T["n_events"], "kernels", len(T["k"]), "moe", len(T["moe"]), flush=True)
    keys = list(rows[0].keys())
    with open(f"{FEAT}/profiles/corrected_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with gzip.open(f"{FEAT}/profiles/inter_moe_gaps_d2_retry2.csv.gz", "wt", newline="") as f:
        w = csv.writer(f); w.writerow(["trace_window", "gap_start_us", "gap_end_us", "gap_len_us", "busy_in_gap_us", "idle_in_gap_us", "ops_in_gap"])
        for k, g in gaps_all.items():
            for x in g: w.writerow([k, x["gap_start_us"], x["gap_end_us"], x["gap_len_us"], x["busy_in_gap_us"], x["idle_in_gap_us"], x["ops_in_gap"]])
    json.dump({"request_window_epoch": [t0 / 1e6, t1 / 1e6], "source_d2_metrics": D2, "traces": d2["trace_files"], "selftest": "gpu_union.py SELFTEST PASS"}, open(f"{FEAT}/profiles/corrected_metrics_meta.json", "w"), indent=1)
    print("done", len(rows), "rows")


if __name__ == "__main__": main()
