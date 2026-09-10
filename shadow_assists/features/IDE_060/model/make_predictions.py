#!/usr/bin/env python3
"""EXP-E11 2단계: held-out 워크로드에 대한 12개 후보의 예측과 각 정책의 선택을 확정한다.
이 스크립트의 출력 (predictions.json) 은 3단계 측정 이전에 커밋되어야 한다 (사전등록).

usage: make_predictions.py <cells.csv> <out_dir> [params.json]
"""
import json, sys, os, random, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import predict, fit, load_cells, N_KV

# ---- 후보 공간 (실행계획서 §17.2) ----
GRAPH_MAX = [160, 192, 224]
SPACING = [16, 32]
CHUNK = [4096, 8192]

def bucket_list(G, s):
    b = list(range(s, G + 1, s))
    if G not in b:
        b.append(G)
    return b

def candidates():
    out = []
    for G, s, ch in itertools.product(GRAPH_MAX, SPACING, CHUNK):
        out.append(dict(id=f"G{G}s{s}c{ch}", kv="fp8", graph_max=G, buckets=bucket_list(G, s),
                        spacing=s, chunk=ch, admission=None, maxtot=143360, mixed=0, cpuinfer=96))
    return out

# ---- held-out 워크로드 (calibration_split.json 과 동일해야 한다) ----
# 동시성은 후보 축이 실제로 드러나는 지점으로 잡는다.
#  WH1: L_avg 768 -> B_cap ~180. C=168 은 (a) 어느 간격에서도 버킷 값이 아니어서 padding 이 다르고
#       (168 -> s16 176 / s32 192), (b) graph_max 160 후보에서는 graph 를 벗어나 eager 가 된다.
#  WH2: L_avg 3200 -> B_cap ~43. C=40 은 prefill 이 전체 토큰의 92% 라 chunk 가 드러나는 지점.
WORKLOADS = [
    dict(id="WH1", dataset="random", Lin=512, Lout=512, C=168, np=504),
    dict(id="WH2", dataset="random", Lin=3072, Lout=256, C=40, np=120),
]

SLO_TPOT_P99 = 300.0   # ms. 예측은 평균 TPOT 이므로 여유를 두고 평균 <= 200 ms 로 건다.
SLO_TPOT_MEAN = 200.0
OVERSUB_MAX = 0.10


def feasible(r):
    return r["oversub"] <= OVERSUB_MAX and r["tpot"] <= SLO_TPOT_MEAN


def policy_cost_model(cands, wl, p):
    ok = [(c, predict(c, wl, p)) for c in cands]
    ok = [(c, r) for c, r in ok if feasible(r)]
    if not ok:
        return None, "feasible 후보 없음"
    c, r = max(ok, key=lambda t: t[1]["tok_s"])
    return c["id"], r


def policy_fixed_epochb(cands, wl, p):
    return "G224s32c4096", predict([c for c in cands if c["id"] == "G224s32c4096"][0], wl, p)


def policy_memory_only(cands, wl, p):
    # KV 용량이 동일하므로 (dtype·max-total-tokens 고정) 남는 판단은 "가장 큰 graph_max".
    c = max(cands, key=lambda c: (c["graph_max"], -c["spacing"]))
    return c["id"], predict(c, wl, p)


def policy_single_factor(cands, wl, p, budget=6):
    """graph_max -> 간격 -> chunk 순서로 한 축씩 고정. 예산 = 실제 측정 횟수 (여기서는 예측으로 대체).
    실제 3단계에서는 이 정책이 고른 순서대로의 측정만 사용해 regret 을 계산한다."""
    cur = dict(graph_max=224, spacing=32, chunk=4096)
    used = []
    for axis, vals in (("graph_max", GRAPH_MAX), ("spacing", SPACING), ("chunk", CHUNK)):
        best, bestv = None, -1
        for v in vals:
            t = dict(cur); t[axis] = v
            cid = f"G{t['graph_max']}s{t['spacing']}c{t['chunk']}"
            c = [x for x in cands if x["id"] == cid][0]
            used.append(cid)
            y = predict(c, wl, p)["tok_s"]
            if y > bestv:
                best, bestv = v, y
        cur[axis] = best
    cid = f"G{cur['graph_max']}s{cur['spacing']}c{cur['chunk']}"
    return cid, predict([x for x in cands if x["id"] == cid][0], wl, p), sorted(set(used))


def policy_random(cands, wl, p, budget=6, seed=42):
    rnd = random.Random(seed)
    picks = rnd.sample(cands, budget)
    c = max(picks, key=lambda c: predict(c, wl, p)["tok_s"])
    return c["id"], predict(c, wl, p), [x["id"] for x in picks]


if __name__ == "__main__":
    cells_csv, out_dir = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        p = json.load(open(sys.argv[3]))
    else:
        cells = load_cells(cells_csv)
        cells = [c for c in cells if c[0].get("dir") != "20260910_002116_expE00_E01_E02"]
        cal = [c for c in cells if c[0]["cpuinfer"] == 96 and c[0]["mixed"] == 0
               and c[0]["cell"] not in ("C1", "C2", "F00", "F10", "A11m64")
               and c[1]["np"] >= 3 * c[1]["C"]]
        p, _ = fit(cal)

    cands = candidates()
    out = dict(params=p, slo=dict(tpot_mean_ms=SLO_TPOT_MEAN, tpot_p99_ms=SLO_TPOT_P99,
                                  oversub_max=OVERSUB_MAX),
               candidates=[dict(id=c["id"], graph_max=c["graph_max"], spacing=c["spacing"],
                                chunk=c["chunk"], buckets=c["buckets"]) for c in cands],
               workloads=WORKLOADS, predictions={}, policy_choice={})
    for wl in WORKLOADS:
        rows = []
        for c in cands:
            r = predict(c, wl, p)
            rows.append(dict(cand=c["id"], tok_s=round(r["tok_s"], 2), tpot=round(r["tpot"], 2),
                             B_eff=round(r["B_eff"], 1), binding=r["binding"],
                             oversub=round(r["oversub"], 3), feasible=feasible(r)))
        out["predictions"][wl["id"]] = rows
        sf = policy_single_factor(cands, wl, p)
        rp = policy_random(cands, wl, p)
        out["policy_choice"][wl["id"]] = dict(
            cost_model=policy_cost_model(cands, wl, p)[0],
            fixed_epochb=policy_fixed_epochb(cands, wl, p)[0],
            memory_only=policy_memory_only(cands, wl, p)[0],
            single_factor=dict(choice=sf[0], probed=sf[2]),
            random_budget6=dict(choice=rp[0], probed=rp[2]))

    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "predictions.json"), "w"), indent=1, ensure_ascii=False)
    for wid, rows in out["predictions"].items():
        print(f"\n=== {wid} ===")
        for r in sorted(rows, key=lambda r: -r["tok_s"]):
            print(f"  {r['cand']:16} pred {r['tok_s']:8.1f} tok/s  TPOT {r['tpot']:6.1f}  "
                  f"B_eff {r['B_eff']:6.1f} ({r['binding']})  oversub {r['oversub']:.2f}  "
                  f"{'ok' if r['feasible'] else 'INFEASIBLE'}")
        print("  정책 선택:", json.dumps(out["policy_choice"][wid], ensure_ascii=False))
