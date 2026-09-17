#!/usr/bin/env python3
"""IDE_074 — 기존 D2 트레이스(TP rank 별)에서 CUDA graph replay 단위로 층별 GPU-side 이벤트를 추출한다.
층 식별은 replay 내 HtoD(Pinned->Device, cold 출력 복사) 의 순번(0..61) 이라는 **패턴 휴리스틱** 이며,
graph 2 (decode bs=63, rows 64) 첫 replay 에서 HtoD 62 / DtoH 248(=62×4) / fused_moe 124(=62×2) 가 확인된 구조를 전제로 한다.
결과 열은 전부 GPU 시계(kineto, host CLOCK_REALTIME 정렬) 의 절대 µs. CPU 이벤트는 만들지 않는다.
  t_dtoh_first_start / t_dtoh_last_end : 층의 CPU 입력 4건 DtoH 첫 시작·마지막 종료
  t_hot_first_start / t_hot_last_end   : 층의 hot MoE 커널 (첫 fused_moe 시작 ~ moe_sum_reduce 종료 = HtoD 직전 커널 종료)
  t_htod_start / t_htod_end            : cold 출력 HtoD (wait_done 이후 첫 GPU 작업)
  t_combine_start / t_combine_end      : HtoD 직후 elementwise(add) 커널
  gap_hot_end_to_htod_us               : HtoD 시작 − 직전 GPU 작업 종료 (같은 stream 이면 done 대기 노출 후보)
  prev_op_name / stream 확인 열 포함. 사용: gpu_layer_timeline.py <trace.json.gz> <out.csv.gz>"""
import gzip, json, sys, csv, collections, os
OPS = ("kernel", "gpu_memcpy", "gpu_memset")


def main(tf, out):
    j = json.load(gzip.open(tf)); base = j.get("baseTimeNanoseconds", 0) / 1e3
    ops = [e for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") in OPS]
    for e in ops: e["_s"] = base + e["ts"]; e["_e"] = e["_s"] + (e.get("dur") or 0)
    ops.sort(key=lambda e: (e["_s"], e.get("args", {}).get("graph node id", 0)))
    ann = [e for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[")]
    for e in ann: e["_s"] = base + e["ts"]; e["_e"] = e["_s"] + (e.get("dur") or 0)
    ann.sort(key=lambda e: e["_s"])
    # replay 분절: graph id 별 node id 감소 지점
    rows = []; last_node = {}; replay_idx = collections.Counter(); cur = collections.defaultdict(list)
    def flush(g):
        seq = cur[g]
        if not seq: return
        htod = [i for i, e in enumerate(seq) if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]]
        # step annotation: replay 의 첫 op 시각을 포함하는 step[...]
        t0 = seq[0]["_s"]; step = next((a for a in ann if a["_s"] <= t0 <= a["_e"]), None)
        if not htod:   # CPU 경로가 없는 rank (tp_rank≠0): all-reduce 커널 순번으로 층 구분 (2개/층 → 짝수 순번 = attention 뒤, 홀수 = MoE 뒤; 휴리스틱)
            ar = [i for i, e in enumerate(seq) if "allreduce" in e["name"].lower() or "all_reduce" in e["name"].lower()]
            for k, i in enumerate(ar):
                e = seq[i]; prev = seq[i - 1] if i > 0 else None
                rows.append({"graph_id": g, "replay": replay_idx[g], "layer_ordinal": k // 2, "allreduce_ordinal": k, "n_allreduce_in_replay": len(ar), "step_name": step["name"] if step else None, "step_start_us": step["_s"] if step else None,
                             "t_allreduce_start": e["_s"], "t_allreduce_end": e["_e"], "allreduce_dur_us": e["_e"] - e["_s"], "allreduce_name": e["name"][:60], "t_prev_op_end": prev["_e"] if prev else None, "prev_op_name": prev["name"][:48] if prev else None,
                             "gap_prev_to_allreduce_us": (e["_s"] - prev["_e"]) if prev else None})
            cur[g] = []; replay_idx[g] += 1; return
        for L, hi in enumerate(htod):
            lo = htod[L - 1] + 1 if L > 0 else 0
            seg = seq[lo:hi]  # 이 층의 HtoD 이전 작업들 (직전 HtoD 이후부터)
            dtoh = [e for e in seg if e["cat"] == "gpu_memcpy" and "DtoH" in e["name"]]
            moe = [e for e in seg if "fused_moe" in e["name"]]
            prev = seq[hi - 1] if hi > 0 else None; h = seq[hi]; nxt = seq[hi + 1] if hi + 1 < len(seq) else None
            allr = [e for e in seq[hi + 1: htod[L + 1] if L + 1 < len(htod) else len(seq)] if "allreduce" in e["name"].lower() or "all_reduce" in e["name"].lower() or "AllReduce" in e["name"]]
            rows.append({"graph_id": g, "replay": replay_idx[g], "layer_ordinal": L, "n_htod_in_replay": len(htod), "step_name": step["name"] if step else None, "step_start_us": step["_s"] if step else None,
                         "t_dtoh_first_start": dtoh[0]["_s"] if dtoh else None, "t_dtoh_last_end": dtoh[-1]["_e"] if dtoh else None, "n_dtoh": len(dtoh), "dtoh_bytes_sum": sum(e.get("args", {}).get("bytes", 0) for e in dtoh),
                         "t_hot_first_start": moe[0]["_s"] if moe else None, "n_fused_moe": len(moe), "hot_kernel_sum_us": sum(e["_e"] - e["_s"] for e in moe),
                         "t_prev_op_end": prev["_e"] if prev else None, "prev_op_name": prev["name"][:48] if prev else None, "prev_op_stream": prev.get("args", {}).get("stream") if prev else None,
                         "t_htod_start": h["_s"], "t_htod_end": h["_e"], "htod_bytes": h.get("args", {}).get("bytes"), "htod_stream": h.get("args", {}).get("stream"),
                         "gap_hot_end_to_htod_us": (h["_s"] - prev["_e"]) if prev else None,
                         "t_combine_start": nxt["_s"] if nxt else None, "t_combine_end": nxt["_e"] if nxt else None, "combine_name": nxt["name"][:40] if nxt else None,
                         "t_allreduce_first_start": allr[0]["_s"] if allr else None, "allreduce_sum_us": sum(e["_e"] - e["_s"] for e in allr), "n_allreduce": len(allr),
                         "seg_ops": len(seg), "seg_busy_sum_us": sum(e["_e"] - e["_s"] for e in seg)})
        cur[g] = []; replay_idx[g] += 1
    for e in ops:
        a = e.get("args", {}); g = a.get("graph id", 0)
        if not g: continue
        n = a.get("graph node id", 0)
        if g in last_node and n <= last_node[g]: flush(g)
        last_node[g] = n; cur[g].append(e)
    for g in list(cur): flush(g)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with gzip.open(out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # 요약
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("gap_hot_end_to_htod_us") is not None: by[("gap_hot_end_to_htod", r["graph_id"], r["n_htod_in_replay"])].append(r["gap_hot_end_to_htod_us"])
        if r.get("allreduce_dur_us") is not None and r.get("allreduce_ordinal", 0) % 2 == 1: by[("allreduce_dur(odd=post-MoE)", r["graph_id"], r["n_allreduce_in_replay"])].append(r["allreduce_dur_us"])
    for k, v in sorted(by.items()):
        v = sorted(v); n = len(v)
        print(f"{k[0]} graph {k[1]} (per replay {k[2]}): n {n} µs p50 {v[n//2]:.1f} p90 {v[int(.9*(n-1))]:.1f} p99 {v[int(.99*(n-1))]:.1f} max {v[-1]:.1f} sum_s {sum(v)/1e6:.3f}")
    if rows and "htod_stream" in rows[0]:
        streams = collections.Counter((r["prev_op_stream"], r["htod_stream"]) for r in rows); print("stream pairs (prev, htod):", streams.most_common(4))
    print("rows", len(rows))


if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
