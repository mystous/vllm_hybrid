#!/usr/bin/env python3
"""IDE_075 §12.5 — TP collective 위치 귀속·rank 별 start/end (CORR 세션의 TP0~3 트레이스, 같은 host 시계).
방법: 각 rank 트레이스에서 graph replay 를 분절(graph id/node id), replay 안의 all-reduce 커널(allreduce_fusion) 을 순번대로 나열하고
직전 커널들에 fused_moe 가 있으면 post_moe, 없으면 post_attention 으로 위치를 귀속한다 (TP0 은 HtoD 순번과 대조해 검증).
rank 별 같은 (replay, ordinal) 의 start 시각 차 = 도착 시차(kineto host 정렬, GPU 간 offset 미검증), duration 은 커널 시간(동기화 포함 가능).
산출: <session>/v2/tp_collective.csv.gz, tp_collective_summary.json"""
import gzip, json, os, sys, csv, collections, statistics as st
HOME = os.path.expanduser("~"); KT = os.environ.get("IDE_KT_HOST", f"{HOME}/.cache/huggingface/kt/ide075")   # IDE_076: 캠페인별 override
OPS = ("kernel", "gpu_memcpy", "gpu_memset")


def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None


def per_rank(tf):
    j = json.load(gzip.open(tf)); base = j.get("baseTimeNanoseconds", 0) / 1e3
    ops = [e for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") in OPS]
    for e in ops: e["_s"] = base + e["ts"]; e["_e"] = e["_s"] + (e.get("dur") or 0)
    ops.sort(key=lambda e: (e["_s"], e.get("args", {}).get("graph node id", 0)))
    ann = sorted(((base + e["ts"], base + e["ts"] + e["dur"], e["name"]) for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[")))
    out = []; last = {}; ridx = collections.Counter(); cur = collections.defaultdict(list)
    def flush(g):
        seq = cur[g]
        if not seq: return
        t0 = seq[0]["_s"]; step = next((a[2] for a in ann if a[0] <= t0 <= a[1]), None); moe_seen = False; k = 0; htod = 0
        for i, e in enumerate(seq):
            nm = e["name"].lower()
            if "fused_moe" in nm: moe_seen = True
            if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]: htod += 1
            if "allreduce" in nm or "all_reduce" in nm:
                out.append({"graph_id": g, "replay": ridx[g], "ordinal": k, "position": "post_moe" if moe_seen else "post_attention", "layer_est": htod if moe_seen else htod, "step": step, "t_start": e["_s"], "t_end": e["_e"], "dur_us": e["_e"] - e["_s"], "name": e["name"][:50]}); k += 1; moe_seen = False
        cur[g] = []; ridx[g] += 1
    for e in ops:
        a = e.get("args", {}); g = a.get("graph id", 0)
        if not g: continue
        n = a.get("graph node id", 0)
        if g in last and n <= last[g]: flush(g)
        last[g] = n; cur[g].append(e)
    for g in list(cur): flush(g)
    return out


def main(sd):
    m = json.load(open(f"{sd}/metrics.json")); boot = m["boot_id"]; prof = next(c for c in m["observer"]["collectors"] if c["collector_id"] == "torch_profiler")
    R = {}
    for n, _ in prof["trace_files"]:
        r = int(n.split("TP-")[1].split(".")[0]); R[r] = per_rank(f"{KT}/{boot}/profiles/{n}")
    key = lambda x: (x["graph_id"], x["replay"], x["ordinal"])
    idx = {r: {key(x): x for x in R[r]} for r in R}
    rows = []
    for k, x0 in idx[0].items():
        row = {"graph_id": k[0], "replay": k[1], "ordinal": k[2], "position": x0["position"], "layer_est": x0["layer_est"], "step": x0["step"]}
        starts = {}
        for r in sorted(R):
            x = idx[r].get(k)
            if not x: row[f"r{r}_missing"] = 1; continue
            row[f"r{r}_start"] = x["t_start"]; row[f"r{r}_dur_us"] = x["dur_us"]; starts[r] = x["t_start"]
            if x["position"] != x0["position"]: row["position_mismatch"] = 1
        if len(starts) == 4: row["arrival_skew_us"] = max(starts.values()) - min(starts.values()); row["last_arrival_rank"] = max(starts, key=starts.get); row["first_arrival_rank"] = min(starts, key=starts.get)
        rows.append(row)
    os.makedirs(f"{sd}/v2", exist_ok=True)
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys: keys.append(kk)
    with gzip.open(f"{sd}/v2/tp_collective.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    S = {}
    for g in sorted(set(r["graph_id"] for r in rows)):
        for pos in ("post_attention", "post_moe"):
            rs = [r for r in rows if r["graph_id"] == g and r["position"] == pos and "arrival_skew_us" in r]
            if not rs: continue
            S[f"{g}:{pos}"] = {"step": rs[0]["step"], "n": len(rs), **{f"r{r}_dur_p50": pct([x[f"r{r}_dur_us"] for x in rs if f"r{r}_dur_us" in x], .5) for r in range(4)}, **{f"r{r}_dur_p95": pct([x[f"r{r}_dur_us"] for x in rs if f"r{r}_dur_us" in x], .95) for r in range(4)},
                               "arrival_skew_p50": pct([x["arrival_skew_us"] for x in rs], .5), "arrival_skew_p95": pct([x["arrival_skew_us"] for x in rs], .95), "last_arrival_rank_counts": dict(collections.Counter(x["last_arrival_rank"] for x in rs)), "first_arrival_rank_counts": dict(collections.Counter(x["first_arrival_rank"] for x in rs)), "position_mismatch": sum(1 for x in rs if x.get("position_mismatch"))}
    json.dump({"session": sd, "by_graph_position": S, "note": "위치 귀속 = 직전 fused_moe 유무 (휴리스틱, TP0 HtoD 순번과 대조); rank 간 시각은 kineto host 정렬, GPU 간 offset 미검증; duration 은 동기화 포함 가능 (링크 시간 아님)"}, open(f"{sd}/v2/tp_collective_summary.json", "w"), indent=1, ensure_ascii=False)
    for k, v in S.items(): print(k, v["step"], "n", v["n"], "dur p50 r0..3", [v[f"r{r}_dur_p50"] for r in range(4)], "skew p50/p95", v["arrival_skew_p50"], v["arrival_skew_p95"], "last", v["last_arrival_rank_counts"], "first", v["first_arrival_rank_counts"])


if __name__ == "__main__": main(sys.argv[1])
