#!/usr/bin/env python3
"""IDE_076 B — calibration 빈도(recorder .pt, decode step 만)로 Hot 집합 재선정 (IDE_070 build_layer_budget 과 같은 greedy: 모든 (layer, expert) 를 빈도순으로 정렬해 상위 S=5,952 슬롯 채움; hotmap 은 층별 빈도 내림차순 → 물리 0..N_l−1 = GPU).
컨테이너에서 실행. 출력: <out>/hotmap.json, layer_budget.json, stats.json (H0 대비 승격/강등 수, 층별 예산 변화, 두 분포에서의 커버리지)
옵션: --decode-only 1 (기본), --min-per-layer/--max-per-layer (기본 0/160), --slots 5952"""
import argparse, json, glob, torch, hashlib, statistics as st
L, E, TOPK = 62, 160, 8


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--freq", required=True); ap.add_argument("--h0-hotmap", required=True); ap.add_argument("--h0-budget", required=True); ap.add_argument("--out", required=True); ap.add_argument("--slots", type=int, default=5952); ap.add_argument("--decode-only", type=int, default=1); ap.add_argument("--min-per-layer", type=int, default=0); ap.add_argument("--max-per-layer", type=int, default=160); ap.add_argument("--prefill-weight", type=float, default=0.0, help="0 = 디코드 빈도만; w>0 이면 f = norm(decode) + w·norm(prefill)")
    a = ap.parse_args(); import os; os.makedirs(a.out, exist_ok=True)
    d = torch.load(a.freq, map_location="cpu", weights_only=False); lc = d["logical_count"]; note = "aggregate"
    if lc.dim() == 3 and a.decode_only:
        ps = lc.sum(-1)[:, 0]; m = (ps > 0) & (ps <= 64 * TOPK); note = f"decode_only steps {int(m.sum())}/{int((ps > 0).sum())}"; lc_dec = lc[m].sum(0).to(torch.float64); lc_pre = lc[ps > 512].sum(0).to(torch.float64)
    else: lc_dec = (lc.sum(0) if lc.dim() == 3 else lc).to(torch.float64); lc_pre = None
    f = lc_dec / lc_dec.sum()
    if lc_pre is not None and a.prefill_weight > 0: f = f + a.prefill_weight * (lc_pre / lc_pre.sum()); note += f" + prefill_weight {a.prefill_weight}"
    # greedy knapsack with per-layer min/max
    per = [a.min_per_layer] * L; order = sorted(((float(f[l, e]), l, e) for l in range(L) for e in range(E)), reverse=True)
    chosen = [set() for _ in range(L)]
    # 층별 상위 min 개 먼저
    for l in range(L):
        for _, e in sorted(((float(f[l, e]), e) for e in range(E)), reverse=True)[:a.min_per_layer]: chosen[l].add(e)
    used = sum(len(c) for c in chosen)
    for v, l, e in order:
        if used >= a.slots: break
        if e in chosen[l] or len(chosen[l]) >= a.max_per_layer: continue
        chosen[l].add(e); used += 1
    per = [len(c) for c in chosen]; assert sum(per) == a.slots, sum(per)
    pm = []
    for l in range(L):
        hot = sorted(chosen[l], key=lambda e: -float(f[l, e])); cold = sorted((e for e in range(E) if e not in chosen[l]), key=lambda e: -float(f[l, e])); pm.append(hot + cold)
    json.dump({"physical_to_logical_map": pm}, open(f"{a.out}/hotmap.json", "w")); json.dump({"slots": a.slots, "slots_used": a.slots, "per_layer": per, "per_layer_min": min(per), "per_layer_max": max(per), "nonuniform": True, "uniform": 96, "source": f"IDE_076 hb_from_freq {note} from {a.freq.split('/')[-1]}"}, open(f"{a.out}/layer_budget.json", "w"), indent=1)
    h0 = json.load(open(a.h0_hotmap))["physical_to_logical_map"]; b0 = json.load(open(a.h0_budget))["per_layer"]
    def cov(pmx, perx, ff): return [float(ff[l][pmx[l][:perx[l]]].sum() / max(float(ff[l].sum()), 1)) for l in range(L)]
    stats = {"freq_note": note, "freq_sha256": sha(a.freq), "hotmap_sha256": sha(f"{a.out}/hotmap.json"), "budget_sha256": sha(f"{a.out}/layer_budget.json"), "per_layer": per, "per_layer_delta_vs_H0": [per[l] - b0[l] for l in range(L)],
             "n_promote": sum(len(chosen[l] - set(h0[l][:b0[l]])) for l in range(L)), "n_demote": sum(len(set(h0[l][:b0[l]]) - chosen[l]) for l in range(L)),
             "coverage_decode": {"HB_mean": st.mean(cov(pm, per, lc_dec)), "HB_min": min(cov(pm, per, lc_dec)), "H0_mean": st.mean(cov(h0, b0, lc_dec)), "H0_min": min(cov(h0, b0, lc_dec))}}
    if lc_pre is not None: stats["coverage_prefill"] = {"HB_mean": st.mean(cov(pm, per, lc_pre)), "H0_mean": st.mean(cov(h0, b0, lc_pre))}
    stats["expected_cold_selections_per_token_decode"] = {"HB": sum((1 - c) * TOPK for c in cov(pm, per, lc_dec)), "H0": sum((1 - c) * TOPK for c in cov(h0, b0, lc_dec))}
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1); print(json.dumps({k: v for k, v in stats.items() if k not in ("per_layer", "per_layer_delta_vs_H0")}, indent=1)); print("per_layer delta:", stats["per_layer_delta_vs_H0"])


if __name__ == "__main__": main()
