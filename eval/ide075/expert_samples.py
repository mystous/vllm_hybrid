#!/usr/bin/env python3
"""IDE_075 A06 — v3 expert 단위 rows 표본 (.experts: slot,epoch,numa,qlen,n,expert:rows;...) 요약. slot 별 1/16 층화(prefill 경로 = qlen>1 작업, 디코드 bs=63 포함).
연결: slot→layer 는 .map (effective_from) 으로. 산출: <boot>/expert_rows_samples.csv.gz, expert_rows_summary.json
표본 percentile 은 표본 분포이며 모집단 percentile 로 확대하지 않음 (sample_probability=1/16 per slot)."""
import gzip, json, os, sys, csv, collections, bisect
HOME = os.path.expanduser("~"); KT = f"{HOME}/.cache/huggingface/kt/ide075"


def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None


def main(bd):
    boot = os.path.basename(bd.rstrip("/")); xp = f"{KT}/{boot}/kt_evt.csv.experts"; mp = f"{KT}/{boot}/kt_evt.csv.map"
    if not os.path.exists(xp): print("NOT_COLLECTED"); return
    smap = collections.defaultdict(list)
    for l in open(mp):
        p = l.strip().split(",")
        if len(p) >= 5: smap[int(p[0])].append((int(p[4]), int(p[1]), int(p[2])))
    for s in smap: smap[s].sort()
    ev = {}
    for r in csv.DictReader(l for l in open(f"{KT}/{boot}/kt_evt.csv") if not l.startswith("#")): ev[(int(r["slot"]), int(r["epoch"]))] = int(r["t_go"])
    rows = []; per_expert = collections.Counter(); per_expert_rows = collections.defaultdict(list)
    for l in open(xp):
        if l.startswith("slot,"): continue
        p = l.strip().split(",")
        if len(p) < 6: continue
        slot, epoch, numa, qlen, n = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]); pairs = [x for x in p[5].split(";") if x]
        tg = ev.get((slot, epoch)); layer = None
        if tg is not None:
            rows_ = smap.get(slot, []); i = bisect.bisect_right([x[0] for x in rows_], tg) - 1
            if i >= 0: layer = rows_[i][1]
        rs = [int(x.split(":")[1]) for x in pairs]; es = [int(x.split(":")[0]) for x in pairs]
        for e, rr in zip(es, rs): per_expert[(layer, e)] += 1; per_expert_rows[(layer, e)].append(rr)
        rows.append({"slot": slot, "epoch": epoch, "layer": layer, "numa": numa, "qlen": qlen, "n_unique": n, "rows_sum": sum(rs), "rows_max": max(rs) if rs else 0, "rows_le2": sum(1 for x in rs if x <= 2), "rows_ge3": sum(1 for x in rs if x >= 3), "pairs": p[5][:400]})
    with gzip.open(f"{bd}/expert_rows_samples.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    big = [r for r in rows if r["qlen"] == 64]; pre = [r for r in rows if r["qlen"] > 64]
    def s(rs, k): return {"n": len(rs), "p50": pct([r[k] for r in rs], .5), "p95": pct([r[k] for r in rs], .95), "max": max((r[k] for r in rs), default=None)}
    allrows = [x for r in big for x in [int(y.split(":")[1]) for y in r["pairs"].split(";") if y]]
    summ = {"samples_total": len(rows), "decode_bs63_samples(qlen64)": len(big), "prefill_samples(qlen>64)": len(pre), "sample_probability": "1/16 per slot (prefill 경로 코드; qlen==1 디코드 미포함)",
            "decode": {"n_unique": s(big, "n_unique"), "rows_max": s(big, "rows_max"), "rows_sum": s(big, "rows_sum"), "expert_rows_distribution": dict(collections.Counter(allrows)), "share_experts_rows_le2": (sum(1 for x in allrows if x <= 2) / len(allrows)) if allrows else None},
            "prefill": {"n_unique": s(pre, "n_unique"), "rows_max": s(pre, "rows_max"), "rows_sum": s(pre, "rows_sum")},
            "top_experts_by_sample_count_decode": [{"layer": k[0], "expert": k[1], "count": v, "rows_p50": pct(per_expert_rows[k], .5)} for k, v in per_expert.most_common(20)]}
    for grp in ("decode", "prefill"):
        for kk, vv in list(summ[grp].items()):
            if isinstance(vv, dict) and vv.get("n") == 0: summ[grp][kk] = {"n": 0}
    json.dump(summ, open(f"{bd}/expert_rows_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps({k: summ[k] for k in ("samples_total", "decode_bs63_samples(qlen64)", "prefill_samples(qlen>64)")}), "decode n_unique p50", summ["decode"]["n_unique"].get("p50"), "share≤2", summ["decode"]["share_experts_rows_le2"])


if __name__ == "__main__": main(sys.argv[1])
