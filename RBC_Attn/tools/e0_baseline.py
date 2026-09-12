#!/usr/bin/env python3
"""E0 — 기준선 복구 (지시서 §11 E0).

대표 세 입력 Nq512/Nk65536/G8/D128/BK128 의 common_private·random·clustered 를
v0.1 과 같은 plan/tile/timing 으로 재현한다. 과거 값을 맞추려고 설정을 고르지 않는다.
재현값이 다르면 version·clock·plan·cache 상태를 기록하고 새 paired baseline 을 고정한다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, kernels, native, planner, runtime  # noqa: E402

POLICIES = ["q_outer", "kv_outer", "signature_only", "rbc"]


def graph_us(fn, replays):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(gr):
            fn()
        rep = gr.replay
    except Exception:
        rep = fn
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(replays):
        rep()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / replays * 1e3


def clocks() -> dict:
    """clock·전력 상태를 기록한다 (설정을 바꾸지 않는다)."""
    q = ("clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,"
         "temperature.gpu,persistence_mode,compute_mode")
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
        return dict(zip(q.split(","), [x.strip() for x in out.split("\n")[0].split(",")]))
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nq", type=int, default=512)
    ap.add_argument("--nk", type=int, default=65536)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--sel", type=int, default=16)
    ap.add_argument("--max-m", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--min-tasks", type=int, default=-1,
                    help="-1 이면 v0.1 의 2*SM (NCU profile 설정 재현)")
    ap.add_argument("--replays", type=int, default=100)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out", default="results/v02/kernel/e0_baseline.json")
    a = ap.parse_args()

    dev = torch.cuda.get_device_properties(0)
    sm = dev.multi_processor_count
    legacy = 2 * sm if a.min_tasks < 0 else a.min_tasks
    rec = dict(meta=dict(
        seed=a.seed, nq=a.nq, nk=a.nk, g=a.g, d=a.d, bk=128, sel=a.sel,
        max_m=a.max_m, legacy_min_tasks=legacy, replays=a.replays, reps=a.reps,
        gpu=dev.name, gpu_uuid=str(getattr(dev, "uuid", None)), sm=sm,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device_count=torch.cuda.device_count(), logical_ordinal=0,
        versions=native.versions(), clocks=clocks(),
        cache_mode="warm_replay (graph 재생, 같은 mask)",
        note="v0.1 과 같은 plan/tile/timing. 과거 값에 맞추려고 설정을 고르지 않았다.",
        time=time.strftime("%Y-%m-%dT%H:%M:%S")), cells=[], errors=[])

    for pattern in ["common_private", "random", "clustered"]:
        s = data.make_synthetic(pattern, hkv=1, nq=a.nq, nk=a.nk, g=a.g, d=a.d, bk=128,
                                sel=a.sel, seed=a.seed)
        ptr, ids = s["ptr"][0], s["block_ids"][0]
        ref_out, ref_lse = data.reference_attention(s, head=0)
        rs = max(1e-6, float(np.abs(ref_out[0]).max()))
        q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
        k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
        v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
        qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
        kw = dict(bk=128, causal=bool(s["causal"]), scale=float(s["scale"]))
        for policy in POLICIES:
            for mm in a.max_m:
                for split, mt in (("none", 0), ("legacy_2SM", legacy)):
                    try:
                        p = planner.plan_head(ptr, ids, bk=128, d=a.d, g=a.g, max_m=mm,
                                              policy=policy, min_tasks=mt)
                        dp = runtime.upload_plan(p, nq=a.nq)
                        out, lse = kernels.run_plan(dp, q, k, v, qpos, **kw)
                        o = out.float().cpu().numpy()
                        rel = float(np.abs(o - ref_out[0]).max() / rs)
                        fin = bool(np.isfinite(o).all())
                        us = [graph_us(lambda: kernels.run_plan(dp, q, k, v, qpos, **kw),
                                       a.replays) for _ in range(a.reps)]
                        st = p.stats(n_edges=len(ids))
                        rec["cells"].append(dict(
                            pattern=pattern, policy=policy, max_m=mm, split=split,
                            min_tasks=mt, gpu_us_median=float(np.median(us)),
                            gpu_us_raw=[float(x) for x in us],
                            rel_err=rel, finite=fin, correct=bool(fin and rel <= 5e-3),
                            n_tasks=st["n_tasks"], partial_slots=st["partial_slots"],
                            plan_ms=p.plan_ms, BM=int(dp.block_m)))
                    except Exception as e:
                        rec["errors"].append(dict(pattern=pattern, policy=policy,
                                                  max_m=mm, split=split,
                                                  detail=f"{type(e).__name__}: {e}"))
        del q, k, v, qpos
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"셀 {len(rec['cells'])}개, 오류 {len(rec['errors'])}건 -> {a.out}")
    for pattern in ["common_private", "random", "clustered"]:
        sub = [c for c in rec["cells"] if c["pattern"] == pattern and c["correct"]]
        if not sub:
            continue
        b = min(sub, key=lambda c: c["gpu_us_median"])
        print(f"  {pattern:16} 최강 {b['policy']:15} M{b['max_m']:<4}{b['split']:11}"
              f"{b['gpu_us_median']:8.1f}us (T={b['n_tasks']})")


if __name__ == "__main__":
    main()
