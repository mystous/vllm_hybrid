#!/usr/bin/env python3
"""IDE_076 B — 비용표·후보 선별 (PLAN.md §8.5~8.8; screening 용, 실제 tok/s 예측 아님).
입력:
  --freq   expert_distribution_recorder_*.pt (logical_count [L,E] 또는 [steps,L,E]; H0 를 만든 calibration 빈도. 출처·해시 기록)
  --costs  producer_layer_costs.csv (v2 분석: 층별 service p50, unique cold p50, cold_pub_delta p50, late_share, hot_span(=overlap_budget 근사) …)
  --hotmap/--budget  현재 H (H0)
  --slope  expert 당 Cold 서비스 한계 비용 µs (측정: replay n 스케일 기울기; R_CPU 확정 후 갱신)
모델 (measured/estimated 출처 표기):
  C_l(H) ≈ slope × E[unique cold experts per job in layer l]   (measured slope; unique 기대값은 빈도로부터: E[unique] = Σ_{e cold} (1 − (1−p_{l,e})^{B}) , B = job 토큰 수 63, p = 층 l 에서 토큰이 expert e 를 고를 확률 (= count/(tokens×topk)×topk ... count 정규화))
  ΔC_l(promote e) = slope × P(e 활성 in job)   (해당 job 에서 e 가 cold 로 등장할 확률만큼 unique 감소)
  ΔG_l(promote k) : hot 커널 비용의 hot 수 의존은 측정 미비 → 'estimated' (S29 hot_span vs budget 회귀 기울기 g1; 표기)
  소비 지연 연결: 층 l 의 cold_pub_delta·late_share 가 클수록 ΔC 가 GPU 대기에 노출됨 → 우선순위 = late_share_l × ΔC
출력: <out>/expert_cost_table.csv (l, e, hot/cold, p_active, P_job, ΔC_us, source), <out>/layer_summary.csv, <out>/candidate_swaps.jsonl (승격/강등 쌍 후보, map diff 는 hotmap_tools.swap 으로 생성)"""
import argparse, csv, glob, json, os, hashlib, math
import torch
L, E, TOPK, B_TOK = 62, 160, 8, 63


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--freq", required=True); ap.add_argument("--costs", required=True); ap.add_argument("--hotmap", required=True); ap.add_argument("--budget", required=True); ap.add_argument("--slope", type=float, required=True); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=8, help="후보당 승격/강등 쌍 수"); ap.add_argument("--decode-only", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = torch.load(a.freq, map_location="cpu", weights_only=False); lc = d["logical_count"]
    freq_note = "aggregate"
    if lc.dim() == 3 and a.decode_only:   # step 별 계수: 층당 선택 수 ≤ 64×8 인 step 만 (디코드 bs ≤ 64) — prefill 청크 step 제외
        per_step = lc.sum(-1)[:, 0]; mask = (per_step > 0) & (per_step <= 64 * TOPK); freq_note = f"decode_only steps {int(mask.sum())}/{int((per_step > 0).sum())} (per-layer selections ≤ {64*TOPK})"; lc = lc[mask]
    lc = lc.sum(0) if lc.dim() == 3 else lc; lc = lc.to(torch.float64); print("freq:", freq_note, "total selections", int(lc.sum()))
    assert lc.shape == (L, E), lc.shape
    p = lc / lc.sum(1, keepdim=True)   # 층 l 에서 한 선택(slot)이 expert e 일 확률; 토큰당 topk 8 선택 → 토큰이 e 를 고를 확률 ≈ 8·p (p 작음)
    pm = json.load(open(a.hotmap))["physical_to_logical_map"]; per = json.load(open(a.budget))["per_layer"]
    hot = [set(pm[l][:per[l]]) for l in range(L)]
    costs = {int(r["producer_layer"]): r for r in csv.DictReader(open(a.costs)) if r["step"].startswith("step[DECODE bs=6")}
    rows = []; layer_rows = []
    for l in range(L):
        pe = (TOPK * p[l]).clamp(max=1.0)                       # 토큰이 expert e 를 고를 확률
        pjob = 1 - (1 - pe) ** B_TOK                            # bs=63 job 에서 e 가 한 번 이상 등장할 확률
        exp_unique_cold = float(sum(pjob[e] for e in range(E) if e not in hot[l]))
        c = costs.get(l, {}); obs_unique = float(c.get("unique_cold_experts_numa0_p50") or 0) if c else None
        layer_rows.append({"layer": l, "budget": per[l], "expected_unique_cold_model": round(exp_unique_cold, 2), "observed_unique_cold_p50_S29": obs_unique, "service_p50_us": c.get("deferred_service_span_p50"), "cold_pub_delta_p50_us": c.get("cold_pub_delta_p50"), "late_share": c.get("late_share"), "overlap_budget_p50_us": c.get("overlap_budget_p50"), "model_service_us": round(a.slope * exp_unique_cold, 1)})
        for e in range(E):
            rows.append({"layer": l, "expert": e, "hot": int(e in hot[l]), "p_select": float(pe[e]), "P_job": float(pjob[e]), "delta_C_us_if_toggled": round(a.slope * float(pjob[e]), 2), "source": "measured_slope(replay n-scale)×calibration_freq(recorder .pt)", "late_share_layer": c.get("late_share")})
    with open(f"{a.out}/expert_cost_table.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f"{a.out}/layer_summary.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys())); w.writeheader(); w.writerows(layer_rows)
    # 후보: 승격 = cold 중 P_job×late_share 가 큰 expert (GPU 대기에 노출된 층), 강등 = hot 중 P_job 이 가장 작은 expert (같은 층 교환 → 층별 슬롯 유지) 또는 CPU 가 이른 층(cold_pub_delta<0) 의 hot
    cands = []
    late = {l: float(costs[l]["late_share"]) if l in costs and costs[l]["late_share"] else 0.0 for l in range(L)}
    early_layers = [l for l in range(L) if l in costs and costs[l]["cold_pub_delta_p50"] and float(costs[l]["cold_pub_delta_p50"]) < 0]
    score = sorted(((late[r["layer"]] * r["delta_C_us_if_toggled"], r) for r in rows if not r["hot"]), key=lambda x: -x[0])
    # 같은 층 교환 후보 (상위 k 승격 각각에 그 층에서 P_job 최소 hot 을 짝지음)
    for s, r in score[:a.k]:
        l = r["layer"]; dm = min((x for x in rows if x["layer"] == l and x["hot"]), key=lambda x: x["P_job"])
        cands.append({"candidate_id": f"SW_L{l}_P{r['expert']}_D{dm['expert']}", "kind": "same_layer_pair", "layer": l, "promote": r["expert"], "demote": dm["expert"], "delta_C_promote_us": r["delta_C_us_if_toggled"], "delta_C_demote_us": -dm["delta_C_us_if_toggled"], "net_delta_C_us_per_job": round(r["delta_C_us_if_toggled"] - dm["delta_C_us_if_toggled"], 2), "late_share": late[l], "priority": round(s, 3), "delta_G": "estimated: 같은 층 슬롯 수 불변 → hot 커널 expert 수 불변 (rows 분포만 변화)", "cost_coverage": "measured slope; freq calibration; ΔG unmeasured"})
    # 층 간 후보: 이른 층(early) 의 최소 P_job hot 를 강등하고 늦은 층의 최대 P_job cold 를 승격
    if early_layers and score:
        for s, r in score[:a.k]:
            l = r["layer"]; best = None
            for le in early_layers:
                dm = min((x for x in rows if x["layer"] == le and x["hot"]), key=lambda x: x["P_job"])
                if best is None or dm["P_job"] < best["P_job"]: best = dm
            if best: cands.append({"candidate_id": f"XL_L{l}_P{r['expert']}_from_L{best['layer']}_D{best['expert']}", "kind": "cross_layer_pair", "layer_promote": l, "promote": r["expert"], "layer_demote": best["layer"], "demote": best["expert"], "delta_C_promote_us": r["delta_C_us_if_toggled"], "delta_C_demote_us": -best["delta_C_us_if_toggled"], "net_delta_C_us_per_job": round(r["delta_C_us_if_toggled"] - best["delta_C_us_if_toggled"], 2), "late_share_promote_layer": late[l], "cold_pub_delta_demote_layer": costs[best["layer"]]["cold_pub_delta_p50"], "priority": round(s, 3), "delta_G": "estimated: 층 간 슬롯 이동 → 승격 층 hot 커널 +1 expert, 강등 층 −1 (비용 미측정)", "cost_coverage": "measured slope; freq calibration; ΔG unmeasured"})
    with open(f"{a.out}/candidate_swaps.jsonl", "w") as f:
        for c in cands: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump({"freq_file": a.freq, "freq_sha256": sha(a.freq), "costs_file": a.costs, "hotmap_sha256": sha(a.hotmap), "budget_sha256": sha(a.budget), "slope_us_per_expert": a.slope, "model": "C_l ≈ slope × E[unique cold]; P_job = 1−(1−8p)^63; ΔG estimated", "freq_note": freq_note, "n_candidates": len(cands)}, open(f"{a.out}/cost_model_provenance.json", "w"), indent=1)
    print("layers", len(layer_rows), "candidates", len(cands)); print("model vs observed unique cold (first 8 layers):", [(x["layer"], x["expected_unique_cold_model"], x["observed_unique_cold_p50_S29"]) for x in layer_rows[:8]])
    for c in cands[:6]: print(c["candidate_id"], "net ΔC/job", c["net_delta_C_us_per_job"], "prio", c["priority"])


if __name__ == "__main__": main()
