#!/usr/bin/env python3
"""IDE_076 C11/C12 — calibration 세션(CALIB 플랜: recorder .pt + CORR 계측) 로부터 비용표·ΔG 회귀·후보 배치를 만든다 (호스트에서 실행; torch 필요한 부분은 컨테이너로 위임).
단계:
 1) CALIB 세션 v2 분석 (dependency_v2/analyze_costs_v2, IDE_KT_HOST=kt/ide076) → producer_layer_costs.csv (층별 service·unique cold·cold_pub_delta·late_share·overlap_budget)
 2) recorder .pt (logical_count[L,E]) → 컨테이너에서 b_cost_model.py 실행 (freq = 새 calibration, costs = 1), slope = --slope)
 3) ΔG 추정: 층별 overlap_budget_p50 (go→hot_ready = attention+hot 커널) 을 budget N_l 에 회귀 → g1 (µs/hot expert) [estimated]
 4) 후보 K 개 생성 (b_cost_model 의 candidate_swaps.jsonl → hotmap_tools swap) → placements/HB_<id>/ + placement_candidate.jsonl
사용: python3 calib_build.py <CALIB boot dir> --slope 62 --k 6 --out eval/results/IDE_076_20260917/calibration/v1"""
import argparse, csv, glob, json, os, subprocess, statistics as st, shutil
HOME = os.path.expanduser("~"); REPO = f"{HOME}/projects/vllm_hybrid"; CAMP = f"{REPO}/eval/results/IDE_076_20260917"; D = f"{HOME}/bin/docker"
KTH = f"{HOME}/.cache/huggingface/kt/ide076"


def sh(c, **kw): return subprocess.run(c, shell=True, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("boot"); ap.add_argument("--slope", type=float, default=62.0); ap.add_argument("--k", type=int, default=6); ap.add_argument("--out", required=True); ap.add_argument("--hotmap", default=f"{CAMP}/placements/H0/hotmap.json"); ap.add_argument("--budget", default=f"{CAMP}/placements/H0/layer_budget.json")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); env = dict(os.environ, IDE_KT_HOST=KTH)
    sessions = sorted(glob.glob(f"{a.boot}/CAL*_CORR/")); assert sessions, "no CAL sessions"
    costs_all = []; recs = []
    for sd in sessions:
        if not os.path.exists(f"{sd}v2/producer_layer_costs.csv"):
            r = sh(f"cd {REPO} && python3 eval/ide075/dependency_v2.py {sd} --v2 > {sd}dependency_v2.log 2>&1 && python3 eval/ide075/analyze_costs_v2.py {sd} > {sd}costs_v2.log 2>&1", env=env); print("analyzed", sd, r.returncode)
        if os.path.exists(f"{sd}v2/producer_layer_costs.csv"): costs_all.append((sd, list(csv.DictReader(open(f"{sd}v2/producer_layer_costs.csv")))))
        recs += sorted(glob.glob(f"{sd}recorder/*.pt"))
    assert recs, "no recorder .pt"; rec = recs[-1]
    # 1) 층별 비용 병합 (bs=63 decode graph, 세션 평균)
    per = {}
    for sd, rows in costs_all:
        for r in rows:
            if not r["step"].startswith("step[DECODE bs=6"): continue
            l = int(r["producer_layer"]); per.setdefault(l, []).append(r)
    merged = []
    for l in sorted(per):
        rr = per[l]; f = lambda k: st.mean(float(x[k]) for x in rr if x.get(k) not in ("", None))
        merged.append({"producer_layer": l, "step": rr[0]["step"], "n": sum(int(x["n"]) for x in rr), "deferred_service_span_p50": f("deferred_service_span_p50"), "unique_cold_experts_numa0_p50": f("unique_cold_experts_numa0_p50"), "cold_pub_delta_p50": f("cold_pub_delta_p50"), "late_share": f("late_share"), "overlap_budget_p50": f("overlap_budget_p50"), "gpu_pre_h2d_gap_p50": f("gpu_pre_h2d_gap_p50")})
    with open(f"{a.out}/calib_layer_costs.csv", "w", newline="") as fo: w = csv.DictWriter(fo, fieldnames=list(merged[0].keys())); w.writeheader(); w.writerows(merged)
    # 3) ΔG 회귀: overlap_budget (go→hot_ready) vs N_l
    budget = json.load(open(a.budget))["per_layer"]; xs = [budget[m["producer_layer"] + 1] if m["producer_layer"] + 1 < 62 else budget[m["producer_layer"]] for m in merged]; ys = [m["overlap_budget_p50"] for m in merged]
    mx, my = st.mean(xs), st.mean(ys); g1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9); g0 = my - g1 * mx
    r2 = 1 - sum((y - (g0 + g1 * x)) ** 2 for x, y in zip(xs, ys)) / max(sum((y - my) ** 2 for y in ys), 1e-9)
    dg = {"model": "overlap_budget_us(consumer layer L = producer+1) ≈ g0 + g1 × N_L (hot experts)", "g0_us": g0, "g1_us_per_hot_expert": g1, "r2": r2, "n_layers": len(xs), "source": "estimated (CALIB CORR 트레이스 회귀; 커널 직접 측정 아님)"}
    json.dump(dg, open(f"{a.out}/delta_g_regression.json", "w"), indent=1); print("ΔG:", json.dumps(dg))
    # 2) b_cost_model (컨테이너)
    cdir = f"{KTH}/calib/v1"; sh(f"{D} exec sgl-kt sh -c 'mkdir -p /models/kt/ide076/calib/v1 && chmod 777 /models/kt/ide076/calib/v1'")
    shutil.copy(rec, f"{cdir}/freq.pt"); shutil.copy(f"{a.out}/calib_layer_costs.csv", f"{cdir}/costs.csv"); shutil.copy(a.hotmap, f"{cdir}/hotmap.json"); shutil.copy(a.budget, f"{cdir}/layer_budget.json")
    sh(f"{D} cp {REPO}/eval/ide076/b_cost_model.py sgl-kt:/tmp/ide076_b_cost_model.py")
    r = sh(f"{D} exec sgl-kt python3 /tmp/ide076_b_cost_model.py --freq /models/kt/ide076/calib/v1/freq.pt --costs /models/kt/ide076/calib/v1/costs.csv --hotmap /models/kt/ide076/calib/v1/hotmap.json --budget /models/kt/ide076/calib/v1/layer_budget.json --slope {a.slope} --k {a.k} --out /models/kt/ide076/calib/v1/model"); print(r.stdout[-1500:], r.stderr[-500:])
    for f in glob.glob(f"{cdir}/model/*"): shutil.copy(f, a.out)
    # 4) 후보 → 배치 파일
    cands = [json.loads(l) for l in open(f"{a.out}/candidate_swaps.jsonl")]; ledger = open(f"{CAMP}/placements/placement_candidate.jsonl", "a")
    for c in cands:
        cid = c["candidate_id"]; od = f"{CAMP}/placements/HB_{cid}"
        if c["kind"] == "same_layer_pair": cmd = f"--promote {c['layer']}:{c['promote']} --demote {c['layer']}:{c['demote']}"
        else: cmd = f"--promote {c['layer_promote']}:{c['promote']} --demote {c['layer_demote']}:{c['demote']} --budget-moves {c['layer_demote']}:{c['layer_promote']}:1"
        r = sh(f"cd {REPO} && python3 eval/ide076/hotmap_tools.py swap {a.hotmap} {a.budget} {od} {cid} {cmd}")
        rec_ = {"candidate_id": cid, "parent_map_sha256": None, "kind": c["kind"], "cpu_cost_basis_variant": "R_CPU(TBD)", "calibration_manifest_sha256": None, "promote_logical_experts": [c.get("promote")], "demote_logical_experts": [c.get("demote")], "logical_slot_count": 5952, "measured_cost_references": ["calib_layer_costs.csv", "replay n-scale slope"], "estimated_cost_references": ["delta_g_regression.json", "freq P_job"], "cost_coverage": c.get("cost_coverage"), "screening_result": {"net_delta_C_us_per_job": c.get("net_delta_C_us_per_job"), "priority": c.get("priority"), "delta_G_est_us": (g1 if c["kind"] == "cross_layer_pair" else 0.0)}, "map_valid": r.returncode == 0, "memory_valid": None, "numerical_valid": None, "lifecycle_valid": None, "selection_result": None, "confirmation_result": None, "decision": "PLANNED", "evidence_paths": [od]}
        if r.returncode == 0: rec_["candidate_map_sha256"] = json.load(open(f"{od}/candidate.json"))["candidate_hotmap_sha256"]
        ledger.write(json.dumps(rec_, ensure_ascii=False) + "\n"); print(cid, "map_valid", r.returncode == 0, r.stdout[-120:].strip() or r.stderr[-160:].strip())
    ledger.close(); print("done", a.out)


if __name__ == "__main__": main()
