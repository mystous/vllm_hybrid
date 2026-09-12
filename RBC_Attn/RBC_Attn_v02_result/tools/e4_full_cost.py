#!/usr/bin/env python3
"""E4 — 전체 준비비 포함 비교 (지시서 §11 E4). 원본 JSON 은 §14.2 스키마를 따른다.

측정 모드:
  run_only          사전 준비된 plan 의 GPU·merge
  cold_init         capacity 할당·pin (JIT 는 별도 필드)
  pool_fresh_host   새 host mask 계획·pack·H2D·GPU·merge·완료
  pool_fresh_gpu    GPU mask 준비 이후의 D2H·계획·H2D·GPU·merge·완료  ← 주 판정
  exact_plan_cache  같은 계획 재사용의 lookup/validation·GPU

`pool_fresh` 는 shape capacity 만 재사용하고 mask 내용을 매번 바꾼다. 같은 mask 를 반복하되
cache 를 끈 값도 따로 남긴다 (`pool_fresh_host_same_mask`).
CPU 시간과 GPU 시간을 더해 wall time 이라 쓰지 않는다 — `wall_us` 는 실제 벽시계다.
측정하지 않은 값은 null 과 사유로 남기고 0 으로 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, kernels, native, planner, runtime, select, slots  # noqa: E402

CAND = {"q_outer": "q_outer", "signature": "signature_only", "bounded_merge": "rbc"}
VERSION = "rbc-v02"


def h(x) -> str:
    a = np.ascontiguousarray(x)
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def base_commit() -> str | None:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "v02", "manifest", "base_commit.txt")
    try:
        return open(p).read().strip().split()[0]
    except Exception:
        return None


def blank(mask_origin, timing_mode) -> dict:
    """§14.2 스키마. 측정하지 않은 자리는 null 로 남는다."""
    return dict(
        version=VERSION, source_commit=base_commit(), gpu_uuid=None,
        mask_origin=mask_origin, timing_mode=timing_mode,
        input_hash=None, mask_hash=None, q_position_hash=None,
        selected_policy=None, plan_hash_before_upload=None, plan_hash_executed=None,
        split_policy=None, actual_tasks=None, kernel_buckets=None,
        direct_rows=None, multi_owner_rows=None, partial_slots=None,
        kv_unique_blocks=None, kv_block_visits=None,
        descriptor_live_bytes=None, descriptor_capacity_bytes=None,
        host_pin_calls_hotpath=None, device_allocation_requests_hotpath=None,
        d2h_bytes=None, h2d_bytes=None, h2d_copy_count=None,
        plan_cache_hit=False, cpu_plan_pack_us=[], dma_activity_us=[],
        main_us=[], merge_us=[], gpu_total_us=[], wall_us=[],
        exact_edge_passed=None, numeric_passed=None, notes=[])


def graph_replay_us(fn, replays):
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
    ok = True
    try:
        with torch.cuda.graph(gr):
            fn()
        rep = gr.replay
    except Exception:
        rep = fn
        ok = False
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(replays):
        rep()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / replays * 1e3, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=["512:16:4", "256:16:8", "512:16:8"],
                    help="nq:sel:g — 적용 영역과 비적용 영역을 함께 넣는다")
    ap.add_argument("--patterns", nargs="*",
                    default=["common_private", "random", "clustered"])
    ap.add_argument("--nk", type=int, default=65536)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--max-m", type=int, default=16)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--replays", type=int, default=100)
    ap.add_argument("--table", default="configs/plan_cost_table.json")
    ap.add_argument("--out", default="results/v02/runtime/e4_full_cost.json")
    a = ap.parse_args()

    dev = torch.cuda.get_device_properties(0)
    uuid = str(getattr(dev, "uuid", None))
    rec = dict(meta=dict(shapes=a.shapes, patterns=a.patterns, nk=a.nk, d=a.d,
                         max_m=a.max_m, reps=a.reps, replays=a.replays,
                         gpu=dev.name, gpu_uuid=uuid, sm=dev.multi_processor_count,
                         device_count=torch.cuda.device_count(), logical_ordinal=0,
                         versions=native.versions(),
                         evaluation_seeds=list(range(101, 101 + a.reps)),
                         time=time.strftime("%Y-%m-%dT%H:%M:%S")),
               records=[], errors=[])

    for shape in a.shapes:
        nq, sel, g = (int(x) for x in shape.split(":"))
        for pattern in a.patterns:
            base = data.make_synthetic(pattern, hkv=1, nq=nq, nk=a.nk, g=g, d=a.d,
                                       bk=128, sel=sel, seed=101)
            q = torch.from_numpy(base["q"][0]).cuda().to(torch.bfloat16)
            k = torch.from_numpy(base["k"][0]).cuda().to(torch.bfloat16)
            v = torch.from_numpy(base["v"][0]).cuda().to(torch.bfloat16)
            qpos_np = np.ascontiguousarray(base["q_positions"], np.int32)
            qpos = torch.from_numpy(qpos_np).cuda()
            scale = float(base["scale"])
            causal = bool(base["causal"])
            ref_out, _ = data.reference_attention(base, head=0)
            rs = max(1e-6, float(np.abs(ref_out[0]).max()))
            in_hash = h(base["q"][0])
            qp_hash = h(qpos_np)

            masks = []
            for sd in range(101, 101 + a.reps):
                s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=a.nk, g=g, d=a.d,
                                        bk=128, sel=sel, seed=sd)
                masks.append((np.ascontiguousarray(s["ptr"][0], np.int64),
                              np.ascontiguousarray(s["block_ids"][0], np.int32)))
            d_masks = [torch.from_numpy(ids).cuda() for _, ids in masks]
            ptr0, ids0 = masks[0]

            lim = slots.ShapeLimits(max_nq=nq, max_tasks=nq * 8 + 128,
                                    max_slots=nq * 16 + 128,
                                    max_task_k=nq * sel * 4 + 1024,
                                    max_red=nq * 16 + 128, g=g, d=a.d, bk=128)

            for cname, pol in CAND.items():
                try:
                    # ---------- cold_init ----------
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    rt = slots.Runtime(lim, n_slots=2)
                    cold_ms = (time.perf_counter() - t0) * 1e3
                    r_cold = blank("n/a", "cold_init")
                    r_cold.update(gpu_uuid=uuid, input_hash=in_hash,
                                  selected_policy=cname, split_policy="none",
                                  descriptor_capacity_bytes=lim.i32_words() * 4
                                  + lim.i64_words() * 8,
                                  wall_us=[cold_ms * 1e3],
                                  notes=["capacity 할당·pin 만. JIT 는 jit_first_us 로 분리"])

                    # JIT·warmup 을 별도 항목으로 분리한다.
                    # 후보마다 커널 variant 가 다르므로 (FULL 여부, merge 유무) 후보별로
                    # warmup 해야 한다. 이것을 안 하면 첫 측정에 컴파일이 섞인다.
                    rt.plan_into(ptr0, ids0, 0, policy=pol, max_m=a.max_m, causal=causal)
                    t0 = time.perf_counter()
                    rt.submit(0, q, k, v, qpos, scale=scale)
                    rt.finish(0)
                    jit_us = (time.perf_counter() - t0) * 1e6
                    t0 = time.perf_counter()
                    for wi in range(3):
                        for sl in (0, 1):
                            rt.plan_into(masks[wi % len(masks)][0], masks[wi % len(masks)][1],
                                         sl, policy=pol, max_m=a.max_m, causal=causal)
                            rt.submit(sl, q, k, v, qpos, scale=scale)
                            rt.finish(sl)
                    warm_us = (time.perf_counter() - t0) * 1e6
                    r_cold["notes"].append(f"jit_first_us={jit_us:.0f}")
                    r_cold["notes"].append(f"warmup_us={warm_us:.0f} (두 slot x 3회, "
                                           "이 후보의 커널 variant 컴파일 포함)")
                    rec["records"].append(r_cold)

                    # ---------- run_only ----------
                    p = planner.plan_head(ptr0, ids0, bk=128, d=a.d, g=g,
                                          max_m=a.max_m, policy=pol, min_tasks=0)
                    dp = runtime.upload_plan(p, nq=nq)
                    out, lse = kernels.run_plan(dp, q, k, v, qpos, bk=128,
                                                causal=causal, scale=scale)
                    o = out.float().cpu().numpy()
                    numeric = bool(np.isfinite(o).all()
                                   and np.abs(o - ref_out[0]).max() / rs <= 5e-3)
                    exact_ok, exact_note = planner.verify_exact_partition(p, ptr0, ids0)
                    st = p.stats(n_edges=len(ids0))
                    tot, gok = graph_replay_us(
                        lambda: kernels.run_plan(dp, q, k, v, qpos, bk=128,
                                                 causal=causal, scale=scale), a.replays)
                    mai, _ = graph_replay_us(
                        lambda: kernels.run_plan(dp, q, k, v, qpos, bk=128, causal=causal,
                                                 scale=scale, stages=("main",)), a.replays)
                    if dp.n_multi_rows or dp.n_empty_rows:
                        mrg, _ = graph_replay_us(
                            lambda: kernels.run_plan(dp, q, k, v, qpos, bk=128,
                                                     causal=causal, scale=scale,
                                                     stages=("merge", "empty")),
                            a.replays)
                    else:
                        mrg = 0.0
                    r = blank("host", "run_only")
                    r.update(gpu_uuid=uuid, input_hash=in_hash, mask_hash=h(ids0),
                             q_position_hash=qp_hash, selected_policy=cname,
                             split_policy="none", actual_tasks=st["n_tasks"],
                             kernel_buckets=1, direct_rows=st["direct_rows"],
                             multi_owner_rows=st["multi_owner_rows"],
                             partial_slots=st["partial_slots"],
                             kv_unique_blocks=int(np.unique(p.task_k).size),
                             kv_block_visits=st["kv_block_visits"],
                             h2d_bytes=int(dp.h2d_bytes),
                             h2d_copy_count=int(dp.h2d_copy_count),
                             main_us=[mai], merge_us=[mrg], gpu_total_us=[tot],
                             exact_edge_passed=bool(exact_ok), numeric_passed=numeric,
                             notes=[f"graph_captured={gok}",
                                    "wall_us 없음: 사전 준비된 plan 의 GPU 시간만 센다",
                                    f"exact_note={exact_note}"])
                    rec["records"].append(r)

                    # ---------- pool_fresh_host (mask 마다 새 계획) ----------
                    walls, cpu, dma, gpu_t, hb, live = [], [], [], [], None, None
                    pre_hash = post_hash = None
                    for i, (ptr_i, ids_i) in enumerate(masks):
                        sl = i % 2
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        meta = rt.plan_into(ptr_i, ids_i, sl, policy=pol,
                                            max_m=a.max_m, causal=causal)
                        rt.submit(sl, q, k, v, qpos, scale=scale)
                        tm = rt.finish(sl)
                        walls.append((time.perf_counter() - t0) * 1e6)
                        cpu.append((tm.plan_call_ms + tm.pack_ms) * 1e3)
                        dma.append(tm.dma_ms * 1e3)
                        gpu_t.append(tm.gpu_ms * 1e3)
                        hb, live = tm.h2d_bytes, meta["live_i32_words"] * 4
                        if i == 0:
                            S = rt.slots[sl]
                            pre_hash = h(S.host_i32[:meta["live_i32_words"]].numpy())
                            post_hash = h(S.dev_i32[:meta["live_i32_words"]].cpu().numpy())
                    info = rt.info()
                    r = blank("host", "pool_fresh_host")
                    r.update(gpu_uuid=uuid, input_hash=in_hash,
                             mask_hash=";".join(h(m[1]) for m in masks[:3]) + ";...",
                             q_position_hash=qp_hash, selected_policy=cname,
                             plan_hash_before_upload=pre_hash,
                             plan_hash_executed=post_hash,
                             split_policy="none", actual_tasks=meta["n_tasks"],
                             kernel_buckets=1, direct_rows=meta["direct_rows"],
                             multi_owner_rows=meta["n_multi_rows"],
                             partial_slots=meta["n_partial_slots"],
                             kv_unique_blocks=None, kv_block_visits=None,
                             descriptor_live_bytes=live,
                             descriptor_capacity_bytes=lim.i32_words() * 4
                             + lim.i64_words() * 8,
                             host_pin_calls_hotpath=info["host_pin_calls_hotpath"],
                             device_allocation_requests_hotpath=info[
                                 "device_allocation_requests_hotpath"],
                             d2h_bytes=None, h2d_bytes=int(hb),
                             h2d_copy_count=1 if meta["full_membership"] else 2,
                             plan_cache_hit=False, cpu_plan_pack_us=cpu,
                             dma_activity_us=dma, gpu_total_us=gpu_t, wall_us=walls,
                             exact_edge_passed=None, numeric_passed=None,
                             notes=["mask 를 매번 바꿨다 (9 평가 seed)",
                                    "kv_unique_blocks/visits=null: descriptor 를 slab 안에서만 "
                                    "다루어 hot path 에 집계를 넣지 않았다",
                                    "d2h_bytes=null: host-origin 이라 D2H 가 없다",
                                    "exact/numeric=null: 이 모드는 시간 측정 전용, 정확성은 "
                                    "run_only 와 pool_fresh_gpu 에서 검사",
                                    f"full_membership={meta['full_membership']}",
                                    f"membership_sent={meta['membership_sent']}"])
                    rec["records"].append(r)

                    # ---------- pool_fresh_host_same_mask (cache 꺼짐) ----------
                    walls2 = []
                    for i in range(a.reps):
                        sl = i % 2
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        rt.plan_into(ptr0, ids0, sl, policy=pol, max_m=a.max_m,
                                     causal=causal)
                        rt.submit(sl, q, k, v, qpos, scale=scale)
                        rt.finish(sl)
                        walls2.append((time.perf_counter() - t0) * 1e6)
                    r = blank("host", "pool_fresh_host_same_mask")
                    r.update(gpu_uuid=uuid, input_hash=in_hash, mask_hash=h(ids0),
                             q_position_hash=qp_hash, selected_policy=cname,
                             split_policy="none", plan_cache_hit=False, wall_us=walls2,
                             notes=["같은 mask 를 반복하되 plan cache 를 쓰지 않았다"])
                    rec["records"].append(r)

                    # ---------- pool_fresh_gpu (주 판정) ----------
                    walls3, d2h, cpu3, dma3, gpu3 = [], [], [], [], []
                    num_ok = True
                    for i, (ptr_i, _) in enumerate(masks):
                        sl = i % 2
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        ids_host = d_masks[i].cpu().numpy()   # GPU-origin mask 의 D2H
                        meta = rt.plan_into(ptr_i, ids_host, sl, policy=pol,
                                            max_m=a.max_m, causal=causal)
                        out_i, lse_i = rt.submit(sl, q, k, v, qpos, scale=scale)
                        tm = rt.finish(sl)
                        walls3.append((time.perf_counter() - t0) * 1e6)
                        d2h.append(d_masks[i].numel() * 4)
                        cpu3.append((tm.plan_call_ms + tm.pack_ms) * 1e3)
                        dma3.append(tm.dma_ms * 1e3)
                        gpu3.append(tm.gpu_ms * 1e3)
                        if i == 0:
                            oo = out_i.float().cpu().numpy()
                            num_ok = bool(np.isfinite(oo).all()
                                          and np.abs(oo - ref_out[0]).max() / rs <= 5e-3)
                    r = blank("gpu", "pool_fresh_gpu")
                    r.update(gpu_uuid=uuid, input_hash=in_hash,
                             mask_hash=";".join(h(m[1]) for m in masks[:3]) + ";...",
                             q_position_hash=qp_hash, selected_policy=cname,
                             split_policy="none", actual_tasks=meta["n_tasks"],
                             kernel_buckets=1, direct_rows=meta["direct_rows"],
                             multi_owner_rows=meta["n_multi_rows"],
                             partial_slots=meta["n_partial_slots"],
                             descriptor_live_bytes=meta["live_i32_words"] * 4,
                             descriptor_capacity_bytes=lim.i32_words() * 4
                             + lim.i64_words() * 8,
                             host_pin_calls_hotpath=info["host_pin_calls_hotpath"],
                             device_allocation_requests_hotpath=info[
                                 "device_allocation_requests_hotpath"],
                             d2h_bytes=int(np.median(d2h)), h2d_bytes=int(hb),
                             h2d_copy_count=1 if meta["full_membership"] else 2,
                             cpu_plan_pack_us=cpu3, dma_activity_us=dma3,
                             gpu_total_us=gpu3, wall_us=walls3,
                             numeric_passed=num_ok,
                             notes=["mask indices 의 D2H 를 계측 안에 포함했다",
                                    "CSR offset 은 host 에 있다는 계약 (indexer 계산은 제외)",
                                    "main/merge 분리는 run_only 에서만 (여기서는 단계 분리 "
                                    "측정이 벽시계를 왜곡하므로 넣지 않았다)"])
                    rec["records"].append(r)

                    # ---------- exact_plan_cache ----------
                    key0 = h(ids0)
                    rt.plan_into(ptr0, ids0, 0, policy=pol, max_m=a.max_m, causal=causal)
                    walls4, guard = [], []
                    for _ in range(a.reps):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        tg = time.perf_counter()
                        hit = (h(ids0) == key0)      # 실제 guard — hot path 시간에 남긴다
                        guard.append((time.perf_counter() - tg) * 1e6)
                        if not hit:
                            raise RuntimeError("plan cache guard 불일치")
                        rt.submit(0, q, k, v, qpos, scale=scale)
                        rt.finish(0)
                        walls4.append((time.perf_counter() - t0) * 1e6)
                    r = blank("host", "exact_plan_cache")
                    r.update(gpu_uuid=uuid, input_hash=in_hash, mask_hash=key0,
                             q_position_hash=qp_hash, selected_policy=cname,
                             split_policy="none", plan_cache_hit=True,
                             cpu_plan_pack_us=guard, wall_us=walls4,
                             notes=["계획을 만들지 않고 mask 일치 guard 만 수행",
                                    "cpu_plan_pack_us 는 guard(해시 비교) 시간"])
                    rec["records"].append(r)

                    del rt
                    torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    rec["errors"].append(dict(shape=shape, pattern=pattern,
                                              candidate=cname,
                                              detail=f"{type(e).__name__}: {e}",
                                              tb=traceback.format_exc()[-900:]))
            del q, k, v, qpos, d_masks
            torch.cuda.empty_cache()
            # 입력 식별을 레코드에 붙인다
            for r in rec["records"]:
                r.setdefault("shape", shape)
                r.setdefault("pattern", pattern)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"레코드 {len(rec['records'])}개, 오류 {len(rec['errors'])}건 -> {a.out}")
    for r in rec["records"]:
        if r["timing_mode"] not in ("run_only", "pool_fresh_gpu", "pool_fresh_host"):
            continue
        w = f"{np.median(r['wall_us']):8.1f}us" if r["wall_us"] else "       -"
        gt = f"{np.median(r['gpu_total_us']):7.1f}us" if r["gpu_total_us"] else "      -"
        cp = f"{np.median(r['cpu_plan_pack_us']):6.0f}us" if r["cpu_plan_pack_us"] else "     -"
        print(f"  {r.get('shape','?'):10}{r.get('pattern','?'):16}"
              f"{r['selected_policy']:14}{r['timing_mode']:26} wall {w} gpu {gt} "
              f"cpu_plan_pack {cp}")
    print("\n분해 (pool_fresh_gpu, 중앙값):")
    for r in rec["records"]:
        if r["timing_mode"] != "pool_fresh_gpu":
            continue
        print(f"  {r.get('shape','?'):10}{r.get('pattern','?'):16}{r['selected_policy']:14}"
              f"cpu_plan_pack {np.median(r['cpu_plan_pack_us']):7.0f}us  "
              f"dma {np.median(r['dma_activity_us']):6.0f}us  "
              f"gpu {np.median(r['gpu_total_us']):7.0f}us  "
              f"wall {np.median(r['wall_us']):8.0f}us  "
              f"desc {r['descriptor_live_bytes']}B  h2d {r['h2d_bytes']}Bx"
              f"{r['h2d_copy_count']}  d2h {r['d2h_bytes']}B")
    for e in rec["errors"][:6]:
        print(f"  오류 {e['shape']} {e['pattern']} {e['candidate']}: {e['detail']}")


if __name__ == "__main__":
    main()
