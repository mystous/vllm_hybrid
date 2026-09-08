#!/usr/bin/env python3
"""PLN_008 M2 — 분할 정책 (오프라인 격자 탐색).

입력: 워크로드 (동시성 C, 컨텍스트 ctx, 공유 prefix 여부), 모델·기계 파라미터 (hybrid_step_model 의 것 그대로),
      GPU 메모리 제약 (H 별 최대 KV 토큰 — 부팅 실측: H64/80: 131072, H96: 49152, H112: 불가).
출력: (H, deferred N, KV 위치) 중 예측 처리량 최대 조합 + 대안 순위 + binding 자원 + 예상 이득.
규율: 모델 수정 금지 (M1 예측과 같은 모델). 정책의 선택은 M2 검증 셀에서 측정 전 등록.

usage: partition_policy.py [--C 32] [--ctx 640] [--shared-prefix 0|1] [--json]
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hybrid_step_model as M

FEASIBLE = {64: dict(kv_tokens=131072, mf=0.92), 80: dict(kv_tokens=131072, mf=0.92), 96: dict(kv_tokens=49152, mf=0.92)}
H_GRID = sorted(FEASIBLE); N_GRID = (0, 2, 4); KV_GRID = ("gpu", "hicache")

def choose(C, ctx, shared_prefix, routing=None):
    routing = routing or M.load_routing()
    rows = []
    for H in H_GRID:
        for N in N_GRID:
            for kv in KV_GRID:
                # hicache 는 공유 prefix 워크로드에서만 의미 (host prefix 읽기 → DDR 간섭); 그 외엔 gpu 와 동일하게 예측
                kv_loc = "hicache_same_socket" if (kv == "hicache" and shared_prefix) else "gpu"
                r = M.tpot_ms(C, H, ctx=ctx, deferred_N=N, kv_loc=kv_loc, kv_tokens=FEASIBLE[H]["kv_tokens"], routing=routing)
                tput = r["B_eff"] / r["tpot_ms"] * 1e3
                rows.append(dict(H=H, N=N, kv=kv, tpot_ms=r["tpot_ms"], step_ms=r["step_ms"], B_eff=r["B_eff"], tok_s=round(tput, 1),
                                 binding=r["binding"], thrash=r["f_retract"] > 0, cpu_exposed_ms=r["cpu_exposed_ms"], gpu_expert_ms=r["gpu_expert_ms"]))
    rows.sort(key=lambda x: -x["tok_s"])
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--C", type=int, default=32); ap.add_argument("--ctx", type=int, default=640)
    ap.add_argument("--shared-prefix", type=int, default=0); ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    rows = choose(a.C, a.ctx, bool(a.shared_prefix))
    if a.json: print(json.dumps(dict(C=a.C, ctx=a.ctx, shared_prefix=a.shared_prefix, ranked=rows[:8]), indent=1)); sys.exit()
    best = rows[0]
    print(f"workload C={a.C} ctx={a.ctx} shared_prefix={a.shared_prefix} → 선택: H={best['H']} N={best['N']} kv={best['kv']} | {best['tok_s']} tok/s, TPOT {best['tpot_ms']} ms, binding {best['binding']}{' THRASH' if best['thrash'] else ''}")
    print(f"{'H':>3s} {'N':>2s} {'kv':8s} {'tok/s':>7s} {'TPOT':>6s} {'Beff':>4s} {'cpuExp':>6s} {'gExp':>5s} binding")
    for r in rows[:10]:
        print(f"{r['H']:3d} {r['N']:2d} {r['kv']:8s} {r['tok_s']:7.1f} {r['tpot_ms']:6.1f} {r['B_eff']:4d} {r['cpu_exposed_ms']:6.1f} {r['gpu_expert_ms']:5.1f} {r['binding']}{' THRASH' if r['thrash'] else ''}")
