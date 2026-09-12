#!/usr/bin/env python3
"""E1 — 최종 계획 선택과 분할 (지시서 §11 E1).

필수 표: policy / split / BM / ntasks / KV_visits / partial_slots / main_us / merge_us /
GPU_total_us / selected_reason
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rbc import data, kernels, planner, runtime

ap = argparse.ArgumentParser()
ap.add_argument("--nq", type=int, default=512); ap.add_argument("--nk", type=int, default=65536)
ap.add_argument("--g", type=int, default=8); ap.add_argument("--d", type=int, default=128)
ap.add_argument("--sel", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--replays", type=int, default=50); ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", default="results/v02/kernel/e1_finalization.json")
a = ap.parse_args()
sm = torch.cuda.get_device_properties(0).multi_processor_count

def graph_us(fn, replays):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    g_ = torch.cuda.CUDAGraph(); side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side): fn()
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g_): fn()
        rep = g_.replay
    except Exception:
        rep = fn
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(replays): rep()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / replays * 1e3

rec = dict(meta=dict(nq=a.nq, nk=a.nk, g=a.g, d=a.d, sel=a.sel, seed=a.seed, sm=sm,
                     gpu=torch.cuda.get_device_name(0), replays=a.replays, reps=a.reps,
                     executor="D (P1+P2+P3: direct output, FULL fast path, BN=BK)",
                     time=time.strftime("%Y-%m-%dT%H:%M:%S")), rows=[])

for pattern in ["common_private", "random", "clustered"]:
    s = data.make_synthetic(pattern, hkv=1, nq=a.nq, nk=a.nk, g=a.g, d=a.d, bk=128,
                            sel=a.sel, seed=a.seed)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    ref_out, _ = data.reference_attention(s, head=0)
    rscale = max(1e-6, float(np.abs(ref_out[0]).max()))
    q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
    k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
    v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
    qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
    kw0 = dict(bk=128, causal=bool(s["causal"]), scale=float(s["scale"]))

    for policy in ["q_outer", "kv_outer", "signature_only", "rbc"]:
        for split, mt in (("none", 0), ("legacy_2SM", 2 * sm)):
            for mm in [16, 32, 64, 128]:
                try:
                    p = planner.plan_head(ptr, ids, bk=128, d=a.d, g=a.g, max_m=mm,
                                          policy=policy, min_tasks=mt)
                    dp = runtime.upload_plan(p, nq=a.nq)
                    out, lse = kernels.run_plan(dp, q, k, v, qpos, **kw0)
                    o = out.float().cpu().numpy()
                    rel = float(np.abs(o - ref_out[0]).max() / rscale)
                    ok = bool(np.isfinite(o).all() and rel <= 5e-3)
                    qs = np.diff(p.task_qptr).astype(np.int64)
                    ks = np.diff(p.task_kptr).astype(np.int64)
                    bm = dp.block_m
                    visits = int(sum(((int(qq) + bm - 1) // bm) * int(kk)
                                     for qq, kk in zip(qs, ks) if qq > 0))
                    tot = [graph_us(lambda: kernels.run_plan(dp, q, k, v, qpos, **kw0), a.replays)
                           for _ in range(a.reps)]
                    mai = [graph_us(lambda: kernels.run_plan(dp, q, k, v, qpos, stages=("main",),
                                                             **kw0), a.replays)
                           for _ in range(a.reps)]
                    mrg = [graph_us(lambda: kernels.run_plan(dp, q, k, v, qpos,
                                                             stages=("merge", "empty"), **kw0),
                                    a.replays)
                           for _ in range(a.reps)] if dp.n_multi_rows or dp.n_empty_rows else [0.0]
                    st = p.stats(n_edges=len(ids))
                    rec["rows"].append(dict(
                        pattern=pattern, policy=policy, split=split, BM=int(bm), max_m=mm,
                        ntasks=st["n_tasks"], KV_visits=visits,
                        kv_block_visits_taskwise=st["kv_block_visits"],
                        partial_slots=st["partial_slots"], slots=st["query_task_incidences"],
                        direct_rows=st["direct_rows"], multi_owner_rows=st["multi_owner_rows"],
                        padded_units=st["padded_units"],
                        main_us=float(np.median(mai)), merge_us=float(np.median(mrg)),
                        GPU_total_us=float(np.median(tot)),
                        plan_ms=p.plan_ms, rel_err=rel, correct=ok))
                except Exception as e:
                    rec["rows"].append(dict(pattern=pattern, policy=policy, split=split,
                                            max_m=mm, error=f"{type(e).__name__}: {e}"))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=1)
print(f"행 {len(rec['rows'])}개 -> {a.out}")
