#!/usr/bin/env python3
"""준비된 query chunk 의 CPU 계획 / GPU 실행 중첩 측정.

설계 §8:
  - 이미 준비된 query 를 chunk 로 나눠 CPU plan 과 GPU 실행을 겹친다.
  - `--routing-source gpu` 면 selected indices 의 D2H 를 포함한다.
    CSR offset/길이는 host 에 이미 있다는 계약이다 (실제 indexer 가 offset 도 GPU 에서
    만든다면 그 비용을 추가해야 한다). Indexer 계산 자체는 포함하지 않는다.
  - 비교 대상은 **whole-query 최강 GPU baseline** 이다. 일부러 직렬화한 chunk baseline 을
    이기는 것은 의미가 없다.
  - CPU/GPU 시간을 합해서 wall time 이라 쓰지 않는다. 실제 벽시계만 센다.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rbc import data, kernels, native, planner, runtime  # noqa: E402

POLICIES = ["q_outer", "kv_outer", "signature_only", "rbc"]


def chunk_csr(ptr, block_ids, lo, hi):
    """query [lo,hi) 의 부분 CSR 를 만든다."""
    sub_ptr = ptr[lo:hi + 1] - ptr[lo]
    sub_ids = block_ids[ptr[lo]:ptr[hi]]
    return np.ascontiguousarray(sub_ptr), np.ascontiguousarray(sub_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="common_private")
    ap.add_argument("--capture")
    ap.add_argument("--nq", type=int, default=512)
    ap.add_argument("--nk", type=int, default=65536)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--sel", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-q", type=int, default=256)
    ap.add_argument("--policy", default="rbc", choices=POLICIES)
    ap.add_argument("--routing-source", default="gpu", choices=["gpu", "host"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-m", type=int, default=32)
    ap.add_argument("--min-tasks", type=int, default=-1)
    ap.add_argument("--out", default="results/pipeline.json")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("CUDA 없음", file=sys.stderr); sys.exit(2)

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    min_tasks = 2 * sm if args.min_tasks < 0 else args.min_tasks
    threads = args.threads if args.threads is not None else int(os.environ.get("RBC_THREADS", 1))

    if args.capture:
        s = data.load_capture(args.capture); src = f"capture:{os.path.basename(args.capture)}"
    else:
        s = data.make_synthetic(args.pattern, hkv=1, nq=args.nq, nk=args.nk, g=args.g,
                                d=args.d, bk=args.bk, sel=args.sel, seed=args.seed)
        src = f"synthetic:{args.pattern}"
    h = 0
    ptr, ids = s["ptr"][h], s["block_ids"][h]
    nq, g, d = s["q"].shape[1], s["q"].shape[2], s["q"].shape[3]
    scale = float(s["scale"])
    dev = "cuda"
    q = torch.from_numpy(s["q"][h]).to(dev, torch.bfloat16)
    k = torch.from_numpy(s["k"][h]).to(dev, torch.bfloat16)
    v = torch.from_numpy(s["v"][h]).to(dev, torch.bfloat16)
    qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).to(dev)

    # routing-source gpu: 선택 indices 가 GPU 에 있고 계획 전에 D2H 해야 한다
    d_ids = torch.from_numpy(np.ascontiguousarray(ids, np.int32)).to(dev) \
        if args.routing_source == "gpu" else None

    kw = dict(bk=s["bk"], causal=bool(s["causal"]), scale=scale)

    # --- 기준선 A: whole-query 최강 GPU baseline (run-only, 계획비 제외) ---
    base = {}
    for pol in POLICIES:
        p = planner.plan_head(ptr, ids, bk=s["bk"], d=d, g=g, max_m=args.max_m,
                              policy=pol, min_tasks=min_tasks, threads=threads)
        dp = runtime.upload_plan(p)
        for _ in range(3):
            kernels.run_plan(dp, q, k, v, qpos, **kw)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.reps):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); kernels.run_plan(dp, q, k, v, qpos, **kw); e1.record()
            torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        base[pol] = float(np.median(ts))
    best_pol = min(base, key=base.get)

    # --- 기준선 B: 직렬 chunk (계획 -> 실행 -> 계획 -> 실행) ---
    bounds = [(i, min(nq, i + args.chunk_q)) for i in range(0, nq, args.chunk_q)]

    def plan_chunk(lo, hi):
        if d_ids is not None:
            _ = d_ids[ptr[lo]:ptr[hi]].cpu().numpy()      # D2H 비용 포함
        sp, si = chunk_csr(ptr, ids, lo, hi)
        return planner.plan_head(sp, si, bk=s["bk"], d=d, g=g, max_m=args.max_m,
                                 policy=args.policy, min_tasks=max(1, min_tasks // len(bounds)),
                                 threads=threads)

    def run_chunk(p, lo, hi):
        dp = runtime.upload_plan(p)
        qp = q[lo:hi].contiguous()
        pos = qpos[lo:hi].contiguous()
        return kernels.run_plan(dp, qp, k, v, pos, **kw)

    serial = []
    for _ in range(args.reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for lo, hi in bounds:
            p = plan_chunk(lo, hi)
            run_chunk(p, lo, hi)
        torch.cuda.synchronize(); serial.append((time.perf_counter() - t0) * 1e3)

    # --- 파이프라인: CPU 계획 스레드와 GPU 실행 중첩 ---
    def pipeline_once():
        qout: "queue.Queue" = queue.Queue(maxsize=2)

        def producer():
            for lo, hi in bounds:
                qout.put((plan_chunk(lo, hi), lo, hi))
            qout.put(None)

        th = threading.Thread(target=producer, daemon=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        th.start()
        while True:
            item = qout.get()
            if item is None:
                break
            p, lo, hi = item
            run_chunk(p, lo, hi)
        torch.cuda.synchronize()
        th.join()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(2):
        pipeline_once()
    pipe = [pipeline_once() for _ in range(args.reps)]

    rec = dict(meta=dict(source=src, nq=nq, nk=s["k"].shape[1], g=g, d=d, bk=s["bk"],
                         chunk_q=args.chunk_q, n_chunks=len(bounds), policy=args.policy,
                         routing_source=args.routing_source, max_m=args.max_m,
                         min_tasks=min_tasks, threads=threads, reps=args.reps,
                         versions=native.versions(),
                         time=time.strftime("%Y-%m-%dT%H:%M:%S")),
               whole_query_gpu_ms=base, whole_query_best=best_pol,
               serial_chunk_ms=float(np.median(serial)), serial_raw=serial,
               pipeline_ms=float(np.median(pipe)), pipeline_raw=pipe)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=1, ensure_ascii=False)

    print(f"\n입력 {src} Nq={nq} Nk={s['k'].shape[1]} G={g} D={d} "
          f"chunk={args.chunk_q}({len(bounds)}개) 정책={args.policy} "
          f"routing={args.routing_source} threads={threads}")
    print(f"{'whole-query GPU (run-only)':38}")
    for pol, msv in sorted(base.items(), key=lambda x: x[1]):
        mark = " <- 최강" if pol == best_pol else ""
        print(f"  {pol:20} {msv:8.3f} ms{mark}")
    print(f"{'직렬 chunk (계획+실행)':38} {np.median(serial):8.3f} ms")
    print(f"{'파이프라인 (CPU/GPU 중첩)':38} {np.median(pipe):8.3f} ms")
    print(f"\n파이프라인 대 whole-query 최강: "
          f"{100*(np.median(pipe)-base[best_pol])/base[best_pol]:+.1f}%")
    print(f"파이프라인 대 직렬 chunk:       "
          f"{100*(np.median(pipe)-np.median(serial))/np.median(serial):+.1f}%")
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
