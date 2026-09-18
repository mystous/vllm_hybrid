#!/usr/bin/env python3
"""IDE_076 C04/C06 — expert 단위 comparator/replay (PLAN.md §5, 부록 B "tile comparator" + "job replay" 의 expert 단계).
컨테이너 sgl-kt 안에서 실행 (CPU 전용, 서버가 내려간 상태에서만: 성능 창 간섭 금지).
- AMXInt4_MOE (Qwen 실제 차원: hidden 6144, intermediate 2560, experts 160, top-k 8) 를 서버와 같은 pool 구성(NUMA 2 서브풀 × 48 스레드)으로 만든다.
- 가중치: 고정 seed 의 bf16 난수 (실제 체크포인트가 아님 — 커널 구조·시간·후보 간 출력 동등성 비교용; 실제 job replay 는 서버 계측/mb480 경로에서 따로).
- 시나리오: 'hot' = 같은 expert 집합·같은 입력을 반복 (cache-hot 커널 시험), 'seq' = 서로 다른 expert 집합의 연속 job (여러 가중치 접근). rows 분포 제어: R=1/2/3/mixed (bs=63 실제 표본 ≈ rows≤2 68 %).
- 출력: <out>/expert_results.csv (시나리오·qlen·n_unique·rows 분포·반복별 forward 경과 ns), <out>/outputs_<scenario>.pt (기준 비교용 출력), <out>/env.json (KT_* env, .so SHA).
사용: python3 expert_comparator.py <out_dir> [--scenarios hot_r1,hot_r2,hot_r3,seq_mixed] [--iters 50] [--threads 96] [--pools 2]"""
import os, sys, time, json, argparse, hashlib, csv
import torch
from kt_kernel import kt_kernel_ext

E, K, H, M = 160, 8, 6144, 2560


def build(threads, pools, seed):
    cfg = kt_kernel_ext.WorkerPoolConfig(); cfg.subpool_count = pools; cfg.subpool_numa_map = list(range(pools)); cfg.subpool_thread_count = [threads // pools + (1 if i < threads % pools else 0) for i in range(pools)]
    ci = kt_kernel_ext.CPUInfer(cfg)
    g = torch.Generator().manual_seed(seed)
    gate = (torch.randn((E, M, H), generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    up = (torch.randn((E, M, H), generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    down = (torch.randn((E, H, M), generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    c = kt_kernel_ext.moe.MOEConfig(E, K, H, M, 0); c.max_len = 8192; c.gate_proj = gate.data_ptr(); c.up_proj = up.data_ptr(); c.down_proj = down.data_ptr(); c.gate_scale = 0; c.pool = ci.backend_
    moe = kt_kernel_ext.moe.AMXInt4_MOE(c); p2l = torch.arange(E, dtype=torch.int64).contiguous()
    ci.submit(moe.load_weights_task(p2l.data_ptr())); ci.sync(); ci.submit(moe.warm_up_task()); ci.sync()
    return ci, moe, (gate, up, down, p2l)


def routing(qlen, n_unique, rows_per_expert, seed):
    """qlen 토큰 × top-k K 의 expert id: nu 개 expert 를 순환 배치해 각 expert 가 정확히 rows_per_expert 행을 받고 토큰 안 expert 는 서로 다르게 (K ≤ nu 필요). nu×rows == qlen×K."""
    assert n_unique * rows_per_expert == qlen * K and K <= n_unique
    g = torch.Generator().manual_seed(seed); experts = torch.randperm(E, generator=g)[:n_unique].tolist()
    ids = torch.tensor([[experts[(t * K + j) % n_unique] for j in range(K)] for t in range(qlen)], dtype=torch.int64).contiguous()
    for t in range(qlen): assert len(set(ids[t].tolist())) == K
    cnt = torch.bincount(ids.flatten(), minlength=E); assert int(cnt.max()) == rows_per_expert and int((cnt > 0).sum()) == n_unique
    return ids, experts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--scenarios", default="hot_r1,hot_r2,hot_r3,seq_mixed"); ap.add_argument("--iters", type=int, default=50); ap.add_argument("--threads", type=int, default=96); ap.add_argument("--pools", type=int, default=2); ap.add_argument("--seed", type=int, default=20260917)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    so = "/usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
    env = {k: v for k, v in os.environ.items() if k.startswith("KT_")}; json.dump({"env": env, "so_sha256": hashlib.sha256(open(so, "rb").read()).hexdigest(), "threads": a.threads, "pools": a.pools, "seed": a.seed}, open(f"{a.out}/env.json", "w"), indent=1)
    ci, moe, W = build(a.threads, a.pools, a.seed)
    rows = []; outs = {}
    for sc in a.scenarios.split(","):
        # 시나리오 정의: (qlen, n_unique, rows_per_expert) — bs=63 디코드는 qlen 64 (63 + capture padding 1 은 실제 routing 에 없음)
        if sc == "hot_r1": qlen, nu, r = 20, 160, 1          # 160 expert × 1 row = 160 = 20 tok × 8
        elif sc == "hot_r2": qlen, nu, r = 40, 160, 2        # 160 × 2 = 320 = 40 × 8
        elif sc == "hot_r3": qlen, nu, r = 60, 160, 3        # 160 × 3 = 480 = 60 × 8
        elif sc == "hot_r1_e18": qlen, nu, r = 2, 16, 1       # 서버 표본과 비슷한 unique 수 (~18): 16 expert × 1 row
        elif sc == "hot_r2_e18": qlen, nu, r = 4, 16, 2
        elif sc.startswith("hot_r1_n"): nu = int(sc[len("hot_r1_n"):]); qlen, r = nu // K, 1        # A2 적용성: expert 수 스케일링 (고정 dispatch 비용 절편) — n8,n16,n32,n64,n128
        elif sc.startswith("hot_r2_n"): nu = int(sc[len("hot_r2_n"):]); qlen, r = nu * 2 // K, 2
        elif sc == "seq_mixed": qlen, nu, r = 64, None, None  # 실제 분포: 무작위 top-8 (rows 1~5 혼합), job 마다 다른 expert 집합
        else: raise SystemExit(f"unknown scenario {sc}")
        g = torch.Generator().manual_seed(a.seed + 1)
        x = (torch.randn((qlen, H), generator=g, dtype=torch.float32) / 100).to(torch.bfloat16).contiguous(); w = (torch.rand((qlen, K), generator=g) / K).contiguous(); out = torch.empty((qlen, H), dtype=torch.bfloat16).contiguous()
        bsz = torch.tensor([qlen], dtype=torch.int64)
        def one(ids):
            t0 = time.perf_counter_ns(); ci.submit(moe.forward_task(bsz.data_ptr(), K, ids.data_ptr(), w.data_ptr(), x.data_ptr(), out.data_ptr(), False)); ci.sync(); return time.perf_counter_ns() - t0
        if sc.startswith("hot"):
            ids, experts = routing(qlen, nu, r, a.seed + 2)
            for _ in range(5): one(ids)
            for it in range(a.iters): ns = one(ids); rows.append({"scenario": sc, "qlen": qlen, "n_unique": nu, "rows_per_expert": r, "iter": it, "elapsed_ns": ns})
            outs[sc] = out.clone()
        else:
            for it in range(a.iters):
                ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(qlen)]).contiguous(); cnt = torch.bincount(ids.flatten(), minlength=E)
                ns = one(ids); rows.append({"scenario": sc, "qlen": qlen, "n_unique": int((cnt > 0).sum()), "rows_per_expert": f"p50={int(cnt[cnt > 0].median())},max={int(cnt.max())}", "iter": it, "elapsed_ns": ns})
                if it == 0: outs[sc] = out.clone(); torch.save(ids, f"{a.out}/ids_{sc}_0.pt")
    with open(f"{a.out}/expert_results.csv", "w", newline="") as f: wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    torch.save({k: v for k, v in outs.items()}, f"{a.out}/outputs.pt")
    import statistics as st
    for sc in a.scenarios.split(","):
        v = sorted(r_["elapsed_ns"] for r_ in rows if r_["scenario"] == sc); print(sc, "n", len(v), "p50 us", v[len(v) // 2] / 1e3, "p10", v[len(v) // 10] / 1e3, "p90", v[int(len(v) * 0.9)] / 1e3, "min", v[0] / 1e3)


if __name__ == "__main__": main()
