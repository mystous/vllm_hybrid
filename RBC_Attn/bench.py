#!/usr/bin/env python3
"""동일 executor 비교 / native 비교 / run-only·fresh 측정.

설계 §8 의 측정 정의:
  gpu_ms        : 미리 계획된 동일 구조를 CUDA graph 로 재생. CPU plan/H2D 불포함.
  fresh_wall_ms : 새 CPU plan + 할당 + descriptor H2D + GPU 본체 + merge + 동기화.
                  mask 는 반복 간 고정하지만 매번 다시 계획한다. JIT 는 warmup 으로 제외.

규율:
  - 모든 내부 정책에 같은 M 탐색 기회를 준다 (max_m 16/32/64/128).
  - min_tasks 는 모든 내부 비교군에 같은 값을 준다 (기본 2×SM).
  - 각 계획의 결과를 먼저 검사한 뒤, 반복은 무작위 순서로 섞어 측정한다.
  - CPU/GPU 시간을 합해서 wall time 이라 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rbc import data, kernels, native, planner, runtime  # noqa: E402

POLICIES = ["q_outer", "kv_outer", "signature_only", "rbc"]


def _torch():
    import torch
    return torch


# --------------------------------------------------------------------------
def load_sample(args):
    if args.capture:
        s = data.load_capture(args.capture)
        s["source"] = f"capture:{os.path.basename(args.capture)}"
        s["synthetic"] = False
    else:
        s = data.make_synthetic(args.pattern, hkv=args.hkv, nq=args.nq, nk=args.nk,
                                g=args.g, d=args.d, bk=args.bk, sel=args.sel,
                                seed=args.seed, causal=not args.no_causal)
        s["source"] = f"synthetic:{args.pattern}"
        s["synthetic"] = True
    return s


def to_device(s, head, dtype):
    torch = _torch()
    td = dict(bfloat16=torch.bfloat16, float16=torch.float16)[dtype]
    q = torch.from_numpy(s["q"][head]).cuda().to(td)
    k = torch.from_numpy(s["k"][head]).cuda().to(td)
    v = torch.from_numpy(s["v"][head]).cuda().to(td)
    qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
    return q, k, v, qpos


# --------------------------------------------------------------------------
def make_plan(s, head, policy, max_m, min_tasks, cost, threads):
    d = s["q"].shape[3]
    g = s["q"].shape[2]
    return planner.plan_head(s["ptr"][head], s["block_ids"][head], bk=s["bk"], d=d, g=g,
                             max_m=max_m, policy=policy, min_tasks=min_tasks,
                             cost=cost, threads=threads)


def time_gpu_graph(dp, q, k, v, qpos, s, reps, ws):
    """CUDA graph 재생으로 run-only 시간을 잰다."""
    torch = _torch()
    kwargs = dict(bk=s["bk"], causal=bool(s["causal"]), scale=float(s["scale"]),
                  workspace=ws)
    for _ in range(3):                                   # warmup (JIT 포함)
        kernels.run_plan(dp, q, k, v, qpos, **kwargs)
    torch.cuda.synchronize()

    g_ = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        kernels.run_plan(dp, q, k, v, qpos, **kwargs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g_):
            kernels.run_plan(dp, q, k, v, qpos, **kwargs)
    except Exception as e:                               # graph 캡처 불가 시 직접 측정
        times = []
        for _ in range(reps):
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record(); kernels.run_plan(dp, q, k, v, qpos, **kwargs); ev1.record()
            torch.cuda.synchronize(); times.append(ev0.elapsed_time(ev1))
        return times, f"graph 미사용: {type(e).__name__}: {e}"

    times = []
    for _ in range(reps):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record(); g_.replay(); ev1.record()
        torch.cuda.synchronize(); times.append(ev0.elapsed_time(ev1))
    return times, None


def time_fresh(s, head, policy, max_m, min_tasks, cost, threads, q, k, v, qpos, reps):
    """매번 새로 계획 + 할당 + H2D + GPU + merge + 동기화."""
    torch = _torch()
    out = dict(wall=[], plan=[], h2d=[])
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = make_plan(s, head, policy, max_m, min_tasks, cost, threads)
        t1 = time.perf_counter()
        dp = runtime.upload_plan(p)
        t2 = time.perf_counter()
        kernels.run_plan(dp, q, k, v, qpos, bk=s["bk"], causal=bool(s["causal"]),
                         scale=float(s["scale"]))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        out["wall"].append((t3 - t0) * 1e3)
        out["plan"].append((t1 - t0) * 1e3)
        out["h2d"].append((t2 - t1) * 1e3)
    return out


def time_native(s, head, q, k, v, reps, scale):
    """native: plan + run 을 fresh 로, run 만 따로 잰다."""
    torch = _torch()
    g, d = s["q"].shape[2], s["q"].shape[3]
    indptr, indices, lpl = native.build_metadata(
        s["ptr"][head], s["block_ids"][head], s["q_positions"], s["bk"], bool(s["causal"]))
    nr = native.NativeRunner()
    nr.plan(indptr, indices, lpl, g=g, d=d, bk=s["bk"], scale=scale)
    for _ in range(3):
        nr.run(q, k, v, bk=s["bk"], scale=scale)
    torch.cuda.synchronize()

    run_only, fresh = [], []
    for _ in range(reps):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record(); nr.run(q, k, v, bk=s["bk"], scale=scale); ev1.record()
        torch.cuda.synchronize(); run_only.append(ev0.elapsed_time(ev1))
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nr2 = native.NativeRunner()
        nr2.plan(indptr, indices, lpl, g=g, d=d, bk=s["bk"], scale=scale)
        nr2.run(q, k, v, bk=s["bk"], scale=scale)
        torch.cuda.synchronize()
        fresh.append((time.perf_counter() - t0) * 1e3)
    out, lse = nr.run(q, k, v, bk=s["bk"], scale=scale)
    return run_only, fresh, out, lse


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="common_private",
                    choices=["common_private", "random", "clustered", "disjoint"])
    ap.add_argument("--capture")
    ap.add_argument("--hkv", type=int, default=1)
    ap.add_argument("--nq", type=int, default=256)
    ap.add_argument("--nk", type=int, default=65536)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--sel", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--fresh", action="store_true", help="fresh wall time 도 측정")
    ap.add_argument("--native", default="optional", choices=["required", "optional", "off"])
    ap.add_argument("--max-m", type=int, nargs="*", default=[16, 32, 64, 128])
    ap.add_argument("--min-tasks", type=int, default=-1,
                    help="-1 이면 2*SM 수. 0 이면 보강 없음")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cost-config")
    ap.add_argument("--out", default="results/bench.json")
    args = ap.parse_args()

    torch = _torch()
    if not torch.cuda.is_available():
        print("CUDA 없음 — 지시서 §1-3 에 따라 이것은 통과가 아니다", file=sys.stderr)
        sys.exit(2)

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    min_tasks = 2 * sm if args.min_tasks < 0 else args.min_tasks
    threads = args.threads if args.threads is not None else int(os.environ.get("RBC_THREADS", 1))
    cost = json.load(open(args.cost_config)) if args.cost_config else None

    s = load_sample(args)
    head = 0
    g, d = s["q"].shape[2], s["q"].shape[3]
    scale = float(s["scale"])
    q, k, v, qpos = to_device(s, head, args.dtype)
    ref_out, ref_lse = data.reference_attention(s, head=head)
    ref_scale = max(1e-6, float(np.abs(ref_out[head]).max()))

    rec = dict(
        meta=dict(source=s["source"], synthetic=bool(s["synthetic"]),
                  nq=int(s["q"].shape[1]), nk=int(s["k"].shape[1]), g=g, d=d,
                  bk=int(s["bk"]), causal=bool(s["causal"]), dtype=args.dtype,
                  n_edges=int(len(s["block_ids"][head])), reps=args.reps,
                  min_tasks=min_tasks, sm_count=sm, rbc_threads=threads,
                  max_m_grid=args.max_m, cost=cost,
                  versions=native.versions(), host=platform.node(),
                  time=time.strftime("%Y-%m-%dT%H:%M:%S")),
        cells=[], order=[], failures=[])

    ws = runtime.Workspace()

    # --- 1단계: 모든 (정책, max_m) 조합의 계획을 만들고 결과를 검사한다 ---
    cand = []
    for pol in POLICIES:
        for mm in args.max_m:
            try:
                p = make_plan(s, head, pol, mm, min_tasks, cost, threads)
                ok, msg = planner.verify_exact_partition(p, s["ptr"][head], s["block_ids"][head]) \
                    if args.nq <= 64 else (True, "검증 생략(대형 입력)")
                if not ok:
                    rec["failures"].append(dict(kind="plan_contract", policy=pol,
                                                max_m=mm, detail=msg))
                    continue
                dp = runtime.upload_plan(p)
                out, lse = kernels.run_plan(dp, q, k, v, qpos, bk=s["bk"],
                                            causal=bool(s["causal"]), scale=scale,
                                            workspace=ws)
                o = out.float().cpu().numpy()
                rel = float(np.abs(o - ref_out[head]).max() / ref_scale)
                fin = np.isfinite(ref_lse[head])
                lerr = float(np.abs(lse.float().cpu().numpy()[fin] - ref_lse[head][fin]).max()) \
                    if fin.any() else 0.0
                if not np.isfinite(o).all():
                    rec["failures"].append(dict(kind="nan_inf", policy=pol, max_m=mm))
                    continue
                cand.append(dict(policy=pol, max_m=mm, plan=p, dplan=dp,
                                 rel_err=rel, lse_err=lerr,
                                 stats=p.stats(n_edges=rec["meta"]["n_edges"]),
                                 plan_ms_once=p.plan_ms))
            except Exception as e:
                rec["failures"].append(dict(kind="exception", policy=pol, max_m=mm,
                                            detail=f"{type(e).__name__}: {e}"))

    if not cand:
        print("검사를 통과한 계획이 없다", file=sys.stderr)
        json.dump(rec, open(args.out, "w"), indent=1, ensure_ascii=False)
        sys.exit(3)

    # --- 2단계: 무작위 순서로 반복 측정 ---
    rng = random.Random(args.seed)
    order = list(range(len(cand)))
    rng.shuffle(order)
    rec["order"] = [f"{cand[i]['policy']}/m{cand[i]['max_m']}" for i in order]

    for i in order:
        c = cand[i]
        gpu_ms, note = time_gpu_graph(c["dplan"], q, k, v, qpos, s, args.reps, ws)
        cell = dict(policy=c["policy"], max_m=c["max_m"], rel_err=c["rel_err"],
                    lse_err=c["lse_err"], stats=c["stats"], graph_note=note,
                    gpu_ms_raw=gpu_ms,
                    gpu_ms=float(np.median(gpu_ms)),
                    gpu_ms_min=float(np.min(gpu_ms)))
        if args.fresh:
            f = time_fresh(s, head, c["policy"], c["max_m"], min_tasks, cost, threads,
                           q, k, v, qpos, max(5, args.reps // 4))
            cell.update(fresh_wall_ms=float(np.median(f["wall"])),
                        fresh_wall_raw=f["wall"],
                        cpu_plan_ms=float(np.median(f["plan"])),
                        h2d_ms=float(np.median(f["h2d"])))
        rec["cells"].append(cell)

    # --- 3단계: native ---
    if args.native != "off":
        try:
            ro, fr, nout, nlse = time_native(s, head, q, k, v, args.reps, scale)
            no = nout.float().cpu().numpy()
            rec["native"] = dict(
                backend="flashinfer.BatchDecodeWithPagedKVCacheWrapper",
                version=native.FLASHINFER_VERSION,
                run_only_ms=float(np.median(ro)), run_only_raw=ro,
                fresh_ms=float(np.median(fr)), fresh_raw=fr,
                rel_err=float(np.abs(no - ref_out[head]).max() / ref_scale),
                lse_err=float(np.abs(nlse.float().cpu().numpy()[np.isfinite(ref_lse[head])]
                                     - ref_lse[head][np.isfinite(ref_lse[head])]).max()),
            )
        except native.NativeUnsupported as e:
            rec["native"] = dict(status="unsupported", detail=str(e))
            if args.native == "required":
                rec["failures"].append(dict(kind="native_required_unsupported", detail=str(e)))
        except Exception as e:
            rec["native"] = dict(status="error", detail=f"{type(e).__name__}: {e}")
            if args.native == "required":
                rec["failures"].append(dict(kind="native_required_error",
                                            detail=f"{type(e).__name__}: {e}"))
    else:
        rec["native"] = dict(status="off")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=1, ensure_ascii=False)

    # --- 요약 출력 ---
    print(f"\n입력 {rec['meta']['source']}  Nq={rec['meta']['nq']} Nk={rec['meta']['nk']} "
          f"G={g} D={d} BK={rec['meta']['bk']} edge={rec['meta']['n_edges']} "
          f"dtype={args.dtype} min_tasks={min_tasks} threads={threads}")
    hdr = f"{'정책':16} {'max_m':>5} {'task':>6} {'partial':>8} {'padded비':>9} {'gpu_ms':>9}"
    if args.fresh:
        hdr += f" {'fresh_ms':>9} {'plan_ms':>8} {'h2d_ms':>7}"
    hdr += f" {'상대오차':>10}"
    print(hdr)
    for c in sorted(rec["cells"], key=lambda c: (c["policy"], c["max_m"])):
        line = (f"{c['policy']:16} {c['max_m']:5d} {c['stats']['n_tasks']:6d} "
                f"{c['stats']['partial_slots']:8d} "
                f"{c['stats'].get('padded_ratio', float('nan')):9.2f} {c['gpu_ms']:9.3f}")
        if args.fresh:
            line += (f" {c.get('fresh_wall_ms', float('nan')):9.3f} "
                     f"{c.get('cpu_plan_ms', float('nan')):8.3f} "
                     f"{c.get('h2d_ms', float('nan')):7.3f}")
        line += f" {c['rel_err']:10.2e}"
        print(line)
    n = rec.get("native", {})
    if "run_only_ms" in n:
        print(f"{'native':16} {'-':>5} {'-':>6} {'-':>8} {'-':>9} {n['run_only_ms']:9.3f} "
              + (f"{n['fresh_ms']:9.3f} {'-':>8} {'-':>7}" if args.fresh else "")
              + f" {n['rel_err']:10.2e}")
    else:
        print(f"native: {n.get('status')} {n.get('detail','')}")
    if rec["failures"]:
        print(f"\n실패 {len(rec['failures'])}건:")
        for f in rec["failures"]:
            print("  ", f)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
