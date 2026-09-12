#!/usr/bin/env python3
"""E2 — P1/P2/P3 의 단독·누적 효과를 정책별로 분리 측정 (지시서 §11 E2).

행 A: v0.1 보존 (legacy min_tasks=2*SM, direct output 없음, BN=BK)
행 B: P1 적용 (강제 분할 제거)
행 C: B + direct output (P2)
행 D: C + BN/late_v/FULL 단계 적용 (P3)

각 행에서 Q-outer / signature / RBC 를 모두 실행한다. RBC D 를 Q-outer A 와만 비교하지 않는다.
"""
import argparse, itertools, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rbc import data, kernels, planner, runtime

POLICIES = ["q_outer", "signature_only", "rbc"]

ap = argparse.ArgumentParser()
ap.add_argument("--patterns", nargs="*", default=["common_private", "random", "clustered"])
ap.add_argument("--nq", type=int, default=512); ap.add_argument("--nk", type=int, default=65536)
ap.add_argument("--g", type=int, default=8); ap.add_argument("--d", type=int, default=128)
ap.add_argument("--sel", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--reps", type=int, default=9); ap.add_argument("--replays", type=int, default=100)
ap.add_argument("--out", default="results/v02/kernel/e2_ablation.json")
a = ap.parse_args()

sm = torch.cuda.get_device_properties(0).multi_processor_count
uuid = torch.cuda.get_device_properties(0).uuid if hasattr(torch.cuda.get_device_properties(0), "uuid") else None

def timed(dp, q, k, v, qpos, s, **kw):
    """graph 재생 100회의 평균. warmup 후 측정."""
    f = lambda: kernels.run_plan(dp, q, k, v, qpos, bk=s["bk"], causal=bool(s["causal"]),
                                 scale=float(s["scale"]), **kw)
    for _ in range(5): f()
    torch.cuda.synchronize()
    g_ = torch.cuda.CUDAGraph(); side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side): f()
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g_): f()
        rep = g_.replay
    except Exception:
        rep = f
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(a.replays): rep()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / a.replays * 1e3      # us

ROWS = [
    ("A_v01",  dict(legacy=True,  direct=False, bn=None,  late_v=False, full=False)),
    ("B_P1",   dict(legacy=False, direct=False, bn=None,  late_v=False, full=False)),
    ("C_P2",   dict(legacy=False, direct=True,  bn=None,  late_v=False, full=False)),
    ("D_P3a",  dict(legacy=False, direct=True,  bn=64,    late_v=False, full=True)),
    ("D_P3b",  dict(legacy=False, direct=True,  bn=None,  late_v=True,  full=True)),
    ("D_P3c",  dict(legacy=False, direct=True,  bn=None,  late_v=False, full=True)),
]
MAXM = [16, 32, 64, 128]

rec = dict(meta=dict(nq=a.nq, nk=a.nk, g=a.g, d=a.d, sel=a.sel, seed=a.seed, sm=sm,
                     gpu=torch.cuda.get_device_name(0), gpu_uuid=str(uuid),
                     replays=a.replays, reps=a.reps,
                     time=time.strftime("%Y-%m-%dT%H:%M:%S")), cells=[], failures=[])

for pattern in a.patterns:
    s = data.make_synthetic(pattern, hkv=1, nq=a.nq, nk=a.nk, g=a.g, d=a.d, bk=128,
                            sel=a.sel, seed=a.seed)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    ref_out, ref_lse = data.reference_attention(s, head=0)
    rscale = max(1e-6, float(np.abs(ref_out[0]).max()))
    q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
    k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
    v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
    qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).cuda()

    cand = []
    for (row, cfg), pol, mm in itertools.product(ROWS, POLICIES, MAXM):
        try:
            mt = 2 * sm if cfg["legacy"] else 0
            p = planner.plan_head(ptr, ids, bk=128, d=a.d, g=a.g, max_m=mm,
                                  policy=pol, min_tasks=mt)
            dp = runtime.upload_plan(p, nq=a.nq)
            kw = dict(bn=cfg["bn"], late_v=cfg["late_v"], allow_full=cfg["full"])
            if not cfg["direct"]:
                # A/B 행은 direct output 없음을 모사: 모든 슬롯을 partial 로 강제
                dp.slot_mode = torch.zeros_like(dp.slot_mode)
                dp.n_partial_slots = dp.n_slots
                # 모든 행을 merge 로 보낸다
                deg = {}
                for si, qq in enumerate(p.task_q): deg.setdefault(int(qq), []).append(si)
                rows_all = sorted(deg)
                rp = np.zeros(len(rows_all) + 1, np.int32); rs = []
                for i, qq in enumerate(rows_all):
                    rs.extend(deg[qq]); rp[i + 1] = len(rs)
                dp.red_ptr = torch.from_numpy(rp).cuda()
                dp.red_slot = torch.from_numpy(np.array(rs, np.int32)).cuda()
                dp.multi_rows = torch.from_numpy(np.array(rows_all, np.int32)).cuda()
                dp.n_multi_rows = len(rows_all)
            out, lse = kernels.run_plan(dp, q, k, v, qpos, bk=128,
                                        causal=bool(s["causal"]), scale=float(s["scale"]), **kw)
            o = out.float().cpu().numpy()
            rel = float(np.abs(o - ref_out[0]).max() / rscale)
            if not np.isfinite(o).all() or rel > 5e-3:
                rec["failures"].append(dict(row=row, policy=pol, max_m=mm, rel=rel,
                                            kind="numeric")); continue
            cand.append((row, pol, mm, cfg, p, dp, kw, rel))
        except Exception as e:
            rec["failures"].append(dict(row=row, policy=pol, max_m=mm,
                                        kind="exception", detail=f"{type(e).__name__}: {e}"))
    rng = random.Random(a.seed)
    for rep in range(a.reps):
        order = list(range(len(cand))); rng.shuffle(order)
        for i in order:
            row, pol, mm, cfg, p, dp, kw, rel = cand[i]
            us = timed(dp, q, k, v, qpos, s, **kw)
            st = p.stats(n_edges=len(ids))
            rec["cells"].append(dict(pattern=pattern, row=row, policy=pol, max_m=mm,
                                     rep=rep, gpu_us=us, rel_err=rel,
                                     n_tasks=st["n_tasks"], partial_slots=int(dp.n_partial_slots),
                                     multi_rows=int(dp.n_multi_rows),
                                     full=bool(dp.full_membership), bn=kw["bn"] or 128,
                                     late_v=kw["late_v"]))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=1)
print(f"셀 {len(rec['cells'])}개, 실패 {len(rec['failures'])}건 -> {a.out}")
