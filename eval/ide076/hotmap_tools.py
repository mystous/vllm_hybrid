#!/usr/bin/env python3
"""IDE_076 B — Hotmap validator / swap generator / diff (PLAN.md §8.3, §8.9, 부록 B).
형식 (실제 loader 계약): hotmap.json = {"physical_to_logical_map": [[logical ids …160], …62]}; 층 l 의 물리 슬롯 0..N_l−1 이 GPU(Hot),
N_l = layer_budget.json["per_layer"][l] (합 5,952). CPU 는 나머지 물리 슬롯의 logical expert 를 계산.
사용:
  validate <hotmap> <budget>                       → 검사 결과 JSON (permutation, 합, 범위, 중복)
  diff <hotmap_a> <budget_a> <hotmap_b> <budget_b>  → 층별 promote/demote 집합
  swap <hotmap> <budget> <out_dir> <cand_id> --promote l:e [l:e …] --demote l:e [l:e …] [--budget-moves l_from:l_to:k …]
      같은 층 교환은 |P_l| == |D_l| 이어야 함; 층 간 이동은 --budget-moves 로 per_layer 를 함께 바꾼다 (합 5,952 유지)."""
import json, sys, os, hashlib, argparse
L, E = 62, 160


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(hm, bd): return json.load(open(hm))["physical_to_logical_map"], json.load(open(bd))


def validate(pm, b):
    per = b["per_layer"]; res = {"layers": len(pm), "experts_per_layer": [len(x) for x in pm][:3], "slots_sum": sum(per), "per_layer_min": min(per), "per_layer_max": max(per), "errors": []}
    if len(pm) != L: res["errors"].append(f"layer count {len(pm)} != {L}")
    if len(per) != L: res["errors"].append(f"budget layer count {len(per)} != {L}")
    if sum(per) != 5952: res["errors"].append(f"slots sum {sum(per)} != 5952")
    for l, row in enumerate(pm):
        if len(row) != E or sorted(row) != list(range(E)): res["errors"].append(f"layer {l}: not a permutation of 0..{E-1} (len {len(row)}, unique {len(set(row))})")
        if not (0 <= per[l] <= E): res["errors"].append(f"layer {l}: budget {per[l]} out of range")
    res["hot_sets_digest"] = hashlib.sha256(json.dumps([sorted(pm[l][:per[l]]) for l in range(L)]).encode()).hexdigest()[:16]
    res["valid"] = not res["errors"]; return res


def hot_sets(pm, per): return [set(pm[l][:per[l]]) for l in range(L)]


def diff(pma, ba, pmb, bb):
    A = hot_sets(pma, ba["per_layer"]); B = hot_sets(pmb, bb["per_layer"]); out = []
    for l in range(L):
        p = sorted(B[l] - A[l]); d = sorted(A[l] - B[l])
        if p or d or ba["per_layer"][l] != bb["per_layer"][l]: out.append({"layer": l, "promote": p, "demote": d, "budget_a": ba["per_layer"][l], "budget_b": bb["per_layer"][l]})
    return {"changed_layers": len(out), "n_promote": sum(len(x["promote"]) for x in out), "n_demote": sum(len(x["demote"]) for x in out), "layers": out}


def swap(pm, b, promote, demote, budget_moves):
    """promote/demote: dict layer -> list of logical ids. budget_moves: list of (l_from, l_to, k). 반환: 새 pm, 새 budget."""
    pm = [list(r) for r in pm]; per = list(b["per_layer"])
    for lf, lt, k in budget_moves: per[lf] -= k; per[lt] += k
    if sum(per) != 5952: raise ValueError(f"slots sum {sum(per)} != 5952 after budget moves")
    for l in range(L):
        P = list(promote.get(l, [])); D = list(demote.get(l, [])); hot = pm[l][:per[l]]; cold = pm[l][per[l]:]
        # 새 hot 집합 = (기존 hot ∖ D) ∪ P, 크기 per[l]
        old_hot = set(pm[l][:b["per_layer"][l]])
        for e in P:
            if e in old_hot: raise ValueError(f"layer {l}: promote {e} already hot")
        for e in D:
            if e not in old_hot: raise ValueError(f"layer {l}: demote {e} not hot")
        new_hot = [e for e in pm[l][:b["per_layer"][l]] if e not in set(D)] + P
        if len(new_hot) != per[l]: raise ValueError(f"layer {l}: new hot size {len(new_hot)} != budget {per[l]} (|P|={len(P)}, |D|={len(D)}, old {b['per_layer'][l]})")
        rest = [e for e in pm[l] if e not in set(new_hot)]
        # cold 순서: 기존 물리 순서 유지 (hot 에서 내려온 D 는 cold 앞쪽에)
        pm[l] = new_hot + rest
        assert sorted(pm[l]) == list(range(E))
    nb = dict(b); nb["per_layer"] = per; nb["slots_used"] = sum(per); nb["per_layer_min"] = min(per); nb["per_layer_max"] = max(per); nb["source"] = f"IDE_076 swap from {b.get('source', '?')}"
    return pm, nb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="*"); ap.add_argument("--promote", nargs="*", default=[]); ap.add_argument("--demote", nargs="*", default=[]); ap.add_argument("--budget-moves", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "validate":
        pm, b = load(a.args[0], a.args[1]); r = validate(pm, b); r["hotmap_sha256"] = sha(a.args[0]); r["budget_sha256"] = sha(a.args[1]); print(json.dumps(r, ensure_ascii=False, indent=1)); sys.exit(0 if r["valid"] else 1)
    if a.cmd == "diff":
        pma, ba = load(a.args[0], a.args[1]); pmb, bb = load(a.args[2], a.args[3]); print(json.dumps(diff(pma, ba, pmb, bb), ensure_ascii=False)[:4000])
    if a.cmd == "swap":
        pm, b = load(a.args[0], a.args[1]); out, cid = a.args[2], a.args[3]; os.makedirs(out, exist_ok=True)
        P = {}; D = {}
        for x in a.promote: l, e = map(int, x.split(":")); P.setdefault(l, []).append(e)
        for x in a.demote: l, e = map(int, x.split(":")); D.setdefault(l, []).append(e)
        moves = [tuple(map(int, x.split(":"))) for x in a.budget_moves]
        npm, nb = swap(pm, b, P, D, moves); v = validate(npm, nb)
        if not v["valid"]: print(json.dumps(v, ensure_ascii=False)); sys.exit(1)
        hp, bp = f"{out}/hotmap.json", f"{out}/layer_budget.json"; json.dump({"physical_to_logical_map": npm}, open(hp, "w")); json.dump(nb, open(bp, "w"), indent=1)
        rec = {"candidate_id": cid, "parent_hotmap_sha256": sha(a.args[0]), "parent_budget_sha256": sha(a.args[1]), "candidate_hotmap_sha256": sha(hp), "candidate_budget_sha256": sha(bp), "promote": P, "demote": D, "budget_moves": moves, "diff": diff(pm, b, npm, nb), "validation": v}
        json.dump(rec, open(f"{out}/candidate.json", "w"), ensure_ascii=False, indent=1); print(json.dumps({k: rec[k] for k in ("candidate_id", "candidate_hotmap_sha256", "candidate_budget_sha256")}), "changed layers", rec["diff"]["changed_layers"])


if __name__ == "__main__": main()
