#!/usr/bin/env python3
"""IDE_070 / TSK_056 — layer 별 비균일 GPU expert 예산 (지시서 C1/C2).

입력: recorder .pt (logical_count [steps?, layers, experts]) 와 hotmap.json (층별 빈도순 physical→logical).
전역 예산 = 슬롯 수 S (기본 96×62 = 5,952 — hot96 과 같은 HBM). 모든 (layer, rank) 쌍을 빈도순으로
정렬해 상위 S 개를 채운다 (greedy knapsack, 슬롯 크기가 층마다 같으므로 비율 = 빈도 그 자체).
층마다 N_l = 그 층에서 채택된 슬롯 수. hotmap 이 층별 빈도 내림차순이므로 "물리 id 0..N_l-1 = GPU" 가
그대로 성립한다 → 서버에는 층별 N_l 만 넘기면 된다.

출력: layer_budget_<S>.json = {"per_layer": [N_0..N_61], "slots": S, "coverage_mean", "cold_per_token", ...}
비교용으로 uniform 96 의 같은 지표도 계산한다.
"""
import glob, json, os, sys
import torch

def load_counts(src):
    files = sorted(glob.glob(os.path.join(src, "expert_distribution_recorder_*.pt")))
    assert files, src
    d = torch.load(files[-1], map_location="cpu", weights_only=False)
    lc = d["logical_count"]
    if lc.dim() == 3:
        lc = lc.sum(0)
    return lc.to(torch.float64)

def metrics(lc, per_layer, topk=8):
    L, E = lc.shape
    srt = torch.sort(lc, dim=1, descending=True).values
    tot = lc.sum(1)
    cov = torch.stack([srt[l, :per_layer[l]].sum() / max(tot[l].item(), 1) for l in range(L)])
    # cold selections per token = Σ_l (1-cov_l) * topk  (각 층 topk 개 선택 중 cold 비율)
    cold_per_tok = float(((1 - cov) * topk).sum())
    return {"coverage_mean": float(cov.mean()), "coverage_min": float(cov.min()),
            "coverage_max": float(cov.max()), "cold_selections_per_token": cold_per_tok,
            "per_layer_coverage": [round(float(x), 5) for x in cov]}

def main():
    src = sys.argv[1]; slots = int(sys.argv[2]) if len(sys.argv) > 2 else 96 * 62
    out = sys.argv[3] if len(sys.argv) > 3 else f"layer_budget_{slots}.json"
    lc = load_counts(src)
    L, E = lc.shape
    srt = torch.sort(lc, dim=1, descending=True).values           # [L, E] 빈도 내림차순
    share = srt / lc.sum(1, keepdim=True).clamp(min=1)            # 층별 정규화 빈도 (활성 확률)
    cand = [(float(share[l, r]), l, r) for l in range(L) for r in range(E)]
    cand.sort(reverse=True)
    per_layer = [0] * L
    for _, l, r in cand[:slots]:
        per_layer[l] = max(per_layer[l], r + 1)
    # greedy 로 뽑힌 (l, r) 이 연속 rank 인지 확인: rank r 이 뽑혔으면 r' < r 도 뽑혔음 (빈도 단조) → prefix 성립
    used = sum(per_layer)
    m_new = metrics(lc, per_layer)
    m_uni = metrics(lc, [slots // L] * L)
    res = {"slots": slots, "slots_used": used, "per_layer": per_layer,
           "per_layer_min": min(per_layer), "per_layer_max": max(per_layer),
           "nonuniform": m_new, "uniform": m_uni, "source": src}
    json.dump(res, open(out, "w"), indent=1)
    print(f"slots {slots} → per-layer min {min(per_layer)} max {max(per_layer)}  (uniform {slots//L})")
    print(f"  coverage mean  uniform {100*m_uni['coverage_mean']:.3f}%  → nonuniform {100*m_new['coverage_mean']:.3f}%")
    print(f"  coverage min   uniform {100*m_uni['coverage_min']:.2f}%  → nonuniform {100*m_new['coverage_min']:.2f}%")
    print(f"  cold sel/token uniform {m_uni['cold_selections_per_token']:.3f} → nonuniform {m_new['cold_selections_per_token']:.3f}")
    print("  per_layer:", per_layer)
    print("→", out)

if __name__ == "__main__":
    main()
