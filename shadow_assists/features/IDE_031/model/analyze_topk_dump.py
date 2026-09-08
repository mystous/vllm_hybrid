#!/usr/bin/env python3
"""deferred 적중 확률: 토큰별 top-8 중 가중치 하위 N 개에 cold expert 가 들어갈 확률 (hot-H = hotmap 층별 빈도 상위 H).
usage: analyze_topk_dump.py <topk_dump.pt> <routing_stats 없이 hotmap.json 경로>"""
import sys, json, torch, collections
dump = torch.load(sys.argv[1], weights_only=False)
hot = json.load(open(sys.argv[2]))["physical_to_logical_map"]   # [62][160]: 물리 idx 순서 = 빈도 내림차순 (hot-H = 앞 H 개 논리 id)
res = {}
for H in (80, 96):
    hotset = [set(hot[l][:H]) for l in range(len(hot))]
    stats = {N: dict(cold_in_low=0, cold_total=0, low_total=0) for N in (1, 2, 4)}
    for layer, ids, w in dump:
        ids = ids.view(-1, ids.shape[-1]); w = w.view(-1, w.shape[-1])
        hs = hotset[layer]
        for t in range(ids.shape[0]):
            order = torch.argsort(w[t])            # 오름차순 (하위 먼저)
            cold_mask = [int(ids[t, j]) not in hs for j in range(ids.shape[1])]
            n_cold = sum(cold_mask)
            for N in stats:
                low = order[:N].tolist()
                stats[N]["cold_in_low"] += sum(cold_mask[j] for j in low)
                stats[N]["cold_total"] += n_cold
                stats[N]["low_total"] += N
    res[H] = {N: dict(p_cold_caught=round(s["cold_in_low"]/max(1,s["cold_total"]),3),   # cold 선택 중 deferred 에 잡히는 비율
                       cold_frac_in_low=round(s["cold_in_low"]/max(1,s["low_total"]),3)) for N, s in stats.items()}
    print(f"H={H}:", res[H])
json.dump(res, open(sys.argv[1].replace(".pt", "_deferred_stats.json"), "w"), indent=1)
