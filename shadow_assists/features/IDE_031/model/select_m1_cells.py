#!/usr/bin/env python3
"""M1 사전 예측 셀 선정: 격자 H×C×L×KV 에서 기측정 셀 제외, 시드 고정 무작위 12개 + 4축 극단 각 1개 강제."""
import itertools, random, json
H = [64, 80, 96]; C = [8, 16, 32, 64]; L = [640, 2000]; KV = ["gpu", "hicache"]
measured = {(80,16,640,"gpu"), (80,32,640,"gpu"), (80,64,640,"gpu"), (96,16,640,"gpu"), (96,32,640,"gpu"), (96,64,640,"gpu")}
grid = [c for c in itertools.product(H, C, L, KV) if c not in measured]
rng = random.Random(20260908)
def pick_with(pred, chosen):
    cands = [c for c in grid if pred(c) and c not in chosen]
    return rng.choice(cands)
chosen = []
for pred in (lambda c: c[1]==8, lambda c: c[1]==64, lambda c: c[2]==2000, lambda c: c[3]=="hicache"):
    chosen.append(pick_with(pred, chosen))
while len(chosen) < 12:
    c = rng.choice(grid)
    if c not in chosen: chosen.append(c)
out = [dict(H=h, C=c, ctx=l, kv=k) for h, c, l, k in chosen]
json.dump(dict(seed=20260908, grid_size=len(grid), cells=out), open("shadow_assists/features/IDE_031/m1_cells.json", "w"), indent=1)
for o in out: print(o)
