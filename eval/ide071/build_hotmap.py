#!/usr/bin/env python3
"""IDE_071 P6 — CALIBRATION recorder .pt → hotmap·층별 예산 (지시서 §10.1, §10.2).
 PL03: 호출 빈도 기반, 총 슬롯 S 고정 (전역 greedy; IDE_070 build_layer_budget 과 동일 정의)
 PL04: 층별 cold 비율 고려 — 층별 예산 N_l 을 '층별 예상 cold 선택 수 (uniform 96 기준)' 에 비례해 재배분한 뒤 층 안에서 빈도순
 PL05: 노출된 CPU 대기시간 감소량 기반 — priority(l,e) = P[expert e 가 B=64 스텝에서 최소 1회 선택] = 1-(1-p_e)^(64·8) ... 근사: 한 스텝에서 distinct cold expert 수 감소 기대값 / 상주 bytes(동일) → 전역 greedy
   (가정: 층당 CPU 시간 ≈ distinct cold expert 수에 비례; expert 당 bytes 동일. 실제 셀로 확인)
 α map: 패스별 토큰 수로 prefill(>64 tok)/decode(≤64) 분리, score = α·p_p + (1−α)·p_d, uniform 96 (IDE_034 build_hotmap_mixed.py 정의)
출력: ~/.cache/huggingface/kt/ide071/{PL03_cal_freq,PL04_cal_coldfrac,PL05_cal_waitreduce}_{hotmap,budget}.json, hotmap_alpha{25,50,75}.json, 통계 json
사용: ~/venv-bench/bin/python build_hotmap.py <recorder_dir> [slots=5952] [B=64]
"""
import glob, json, os, sys
import torch

def main():
    src = sys.argv[1]; S = int(sys.argv[2]) if len(sys.argv) > 2 else 5952; B = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    outd = os.path.expanduser("~/.cache/huggingface/kt/ide071"); os.makedirs(outd, exist_ok=True)
    fs = sorted(glob.glob(f"{src}/expert_distribution_recorder_*.pt"))
    d = torch.load(fs[-1], map_location="cpu", weights_only=False); lc = d["logical_count"].to(torch.float64)  # [passes, L, E]
    passes_tok = lc.sum(2)[:, 0] / 8.0
    pre = lc[passes_tok > 64].sum(0); dec = lc[passes_tok <= 64].sum(0); tot = lc.sum(0)
    L, E = tot.shape
    def probs(m): return m / m.sum(1, keepdim=True).clamp(min=1)
    p_all, p_p, p_d = probs(tot), probs(pre), probs(dec)
    stats = {"source": os.path.basename(fs[-1]), "passes": int(lc.shape[0]), "prefill_passes": int((passes_tok > 64).sum()), "decode_passes": int((passes_tok <= 64).sum()),
             "prefill_tokens": float(pre[0].sum() / 8), "decode_tokens": float(dec[0].sum() / 8), "slots": S, "B": B, "maps": {}}
    def order_by(score):  # [L,E] → 층별 내림차순 논리 id
        return torch.argsort(score, dim=1, descending=True).tolist()
    def budget_global(score, S):
        cand = sorted(((float(score[l, e]), l, e) for l in range(L) for e in range(E)), reverse=True)[:S]
        per = [0] * L
        for _, l, e in cand: per[l] += 1
        return per
    def metrics(pmap, per, p):
        cov = [float(sum(p[l][e] for e in pmap[l][:per[l]])) for l in range(L)]
        cold_tok = sum((1 - c) * 8 for c in cov)
        # 스텝(B 토큰) 당 기대 distinct cold expert 수 (층 합)
        edc = sum(float(sum(1 - (1 - min(1.0, float(p[l][e]) * 8)) ** B for e in pmap[l][per[l]:])) for l in range(L))
        return {"coverage_mean": sum(cov) / L, "coverage_min": min(cov), "cold_routes_per_token_all_layers": cold_tok, "expected_distinct_cold_per_step_all_layers": edc, "per_layer": per}
    uni = [S // L] * L
    # PL03 freq
    m03 = order_by(p_all); b03 = budget_global(p_all, S)
    # PL04 cold-fraction: 층별 cold 기대량 (uniform 96 기준) 에 비례해 재배분, 최소 32 최대 160
    coldu = torch.tensor([float(sum(p_all[l][e] for e in m03[l][96:])) for l in range(L)])
    w = coldu / coldu.sum(); per04 = [int(round(float(x) * S)) for x in w]
    per04 = [min(160, max(32, x)) for x in per04]; diff = S - sum(per04)
    order = sorted(range(L), key=lambda l: -float(w[l]))
    i = 0
    while diff != 0:
        l = order[i % L]; step = 1 if diff > 0 else -1
        if 32 <= per04[l] + step <= 160: per04[l] += step; diff -= step
        i += 1
    # PL05 wait-reduce: priority = P[e 가 B 스텝에서 선택됨] = 1-(1-8p)^B (distinct cold 감소 기대), 전역 greedy
    pr05 = 1 - (1 - (p_all * 8).clamp(max=1.0)) ** B
    m05 = order_by(pr05); b05 = budget_global(pr05, S)
    maps = {"PL03_cal_freq": (m03, b03, p_all), "PL04_cal_coldfrac": (m03, per04, p_all), "PL05_cal_waitreduce": (m05, b05, p_all)}
    for name, (pmap, per, p) in maps.items():
        json.dump({"physical_to_logical_map": pmap}, open(f"{outd}/{name}_hotmap.json", "w"))
        json.dump({"per_layer": per, "slots": S, "source": stats["source"]}, open(f"{outd}/{name}_budget.json", "w"), indent=1)
        stats["maps"][name] = {"nonuniform": metrics(pmap, per, p), "uniform96_same_order": metrics(pmap, uni, p), "per_layer_min": min(per), "per_layer_max": max(per), "sum": sum(per)}
    for a in (0.25, 0.5, 0.75):
        sc = a * p_p + (1 - a) * p_d; m = order_by(sc)
        json.dump({"physical_to_logical_map": m}, open(f"{outd}/hotmap_alpha{int(a*100)}.json", "w"))
        stats["maps"][f"alpha{int(a*100)}"] = {"uniform96": {"cov_prefill": metrics(m, uni, p_p)["coverage_mean"], "cov_decode": metrics(m, uni, p_d)["coverage_mean"], "edc_decode": metrics(m, uni, p_d)["expected_distinct_cold_per_step_all_layers"]}}
    json.dump(stats, open(f"{outd}/calibration_stats.json", "w"), indent=1)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else {a: b for a, b in vv.items() if a != "per_layer"}) for kk, vv in v.items()} for k, v in stats["maps"].items()}, indent=1)[:3000])

if __name__ == "__main__":
    main()
