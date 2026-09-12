#!/usr/bin/env python3
"""§13.1 게이트 영역 탐색 — non-Q RBC 가 최강 공통 Q-outer 를 5% 이상 줄이는 영역이 있는가.

지시서 §6.4 에 따라 타일 후보는 max_m={16,32,64,128} 로 고정, G/D/BK 등 §1.1 고정 조건 유지.
정책마다 같은 탐색 예산(4 후보)을 주고 각 정책의 최적값을 비교한다.
정확성은 같은 입력의 q_outer(M=16) 출력을 기준으로 상호 대조한다 (대표 입력에서 FP64
reference 로 검증된 경로).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rbc import data, kernels, planner, runtime

ap = argparse.ArgumentParser()
ap.add_argument("--nq", nargs="*", type=int, default=[512, 2048, 8192])
ap.add_argument("--sel", nargs="*", type=int, default=[8, 16, 32])
ap.add_argument("--patterns", nargs="*", default=["common_private", "random", "clustered"])
ap.add_argument("--seeds", nargs="*", type=int, default=[101])
ap.add_argument("--nk", type=int, default=65536)
ap.add_argument("--g", type=int, default=8)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--gs", nargs="*", type=int, default=None, help="G 축 (주면 --g 무시)")
ap.add_argument("--replays", type=int, default=30); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default="results/v02/kernel/gate_region_sweep.json")
a = ap.parse_args()
MAXM = [16, 32, 64, 128]
POL = ["q_outer", "signature_only", "rbc"]

def graph_us(fn, replays):
    for _ in range(3): fn()
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

def plan_sig(p):
    """계획의 구조 지문. q_outer 와 같으면 Q-fallback 으로 본다."""
    return (int(p.n_tasks), hash(p.task_qptr.tobytes()), hash(p.task_q.tobytes()),
            hash(p.task_kptr.tobytes()), hash(p.task_k.tobytes()))

rec = dict(meta=dict(nk=a.nk, g=a.g, d=a.d, bk=128, max_m_candidates=MAXM,
                     gpu=torch.cuda.get_device_name(0),
                     sm=torch.cuda.get_device_properties(0).multi_processor_count,
                     replays=a.replays, reps=a.reps, seeds=a.seeds,
                     time=time.strftime("%Y-%m-%dT%H:%M:%S")), cells=[], errors=[])

GS = a.gs if a.gs else [a.g]
for seed in a.seeds:
  for nq in a.nq:
   for gg in GS:
    for sel in a.sel:
      for pattern in a.patterns:
        try:
            s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=a.nk, g=gg, d=a.d, bk=128,
                                    sel=sel, seed=seed)
        except Exception as e:
            rec["errors"].append(dict(nq=nq, g=gg, sel=sel, pattern=pattern, seed=seed,
                                      stage="data", detail=f"{type(e).__name__}: {e}")); continue
        ptr, ids = s["ptr"][0], s["block_ids"][0]
        q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
        k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
        v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
        qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
        kw = dict(bk=128, causal=bool(s["causal"]), scale=float(s["scale"]))
        ref = None; qsigs = set()
        for pol in POL:
            for mm in MAXM:
                try:
                    p = planner.plan_head(ptr, ids, bk=128, d=a.d, g=gg, max_m=mm,
                                          policy=pol, min_tasks=0)
                    dp = runtime.upload_plan(p, nq=nq)
                    out, lse = kernels.run_plan(dp, q, k, v, qpos, **kw)
                    o = out.float()
                    if pol == "q_outer" and mm == 16:
                        ref = o.clone()
                    rscale = max(1e-6, float(ref.abs().max())) if ref is not None else 1.0
                    rel = float((o - ref).abs().max() / rscale) if ref is not None else -1.0
                    if not torch.isfinite(o).all() or (ref is not None and rel > 5e-3):
                        rec["errors"].append(dict(nq=nq, g=gg, sel=sel, pattern=pattern,
                                                  seed=seed, policy=pol, max_m=mm,
                                                  stage="numeric", rel=rel)); continue
                    sig = plan_sig(p)
                    if pol == "q_outer": qsigs.add(sig)
                    us = float(np.median([graph_us(
                        lambda: kernels.run_plan(dp, q, k, v, qpos, **kw), a.replays)
                        for _ in range(a.reps)]))
                    st = p.stats(n_edges=len(ids))
                    bm = dp.block_m
                    qsz = np.diff(p.task_qptr).astype(np.int64)
                    ksz = np.diff(p.task_kptr).astype(np.int64)
                    visits = int(sum(((int(x) + bm - 1) // bm) * int(y)
                                     for x, y in zip(qsz, ksz) if x > 0))
                    rec["cells"].append(dict(
                        seed=seed, nq=nq, g=gg, sel=sel, pattern=pattern, policy=pol, max_m=mm,
                        BM=int(bm), gpu_us=us, ntasks=st["n_tasks"], KV_visits=visits,
                        padded_units=st["padded_units"], partial_slots=st["partial_slots"],
                        plan_sig=str(sig), plan_ms=p.plan_ms, rel_err=rel))
                except Exception as e:
                    rec["errors"].append(dict(nq=nq, g=gg, sel=sel, pattern=pattern,
                                              seed=seed, policy=pol, max_m=mm, stage="run",
                                              detail=f"{type(e).__name__}: {e}"))
        for c in rec["cells"]:
            if (c["seed"], c["nq"], c["g"], c["sel"], c["pattern"]) == (seed, nq, gg, sel, pattern):
                c["is_q_fallback"] = c["plan_sig"] in {str(x) for x in qsigs}
        del q, k, v, qpos; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=1)
print(f"셀 {len(rec['cells'])}개, 오류/제외 {len(rec['errors'])}건 -> {a.out}")
