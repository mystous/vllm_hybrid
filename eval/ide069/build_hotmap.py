#!/usr/bin/env python3
"""recorder .pt (logical_count [layers, experts]) → hotmap.json (physical_to_logical_map).

층마다 호출 빈도 내림차순으로 logical expert 를 정렬해 물리 id 0 부터 배치한다.
KT 는 `--kt-num-gpu-experts N` 으로 물리 id 0..N-1 을 GPU 에 올리므로, 이 재배열이
"빈도 상위 N 개가 GPU" 를 만든다. 가중치 파일은 건드리지 않는다.
커버리지(상위 N 개가 전체 호출의 몇 %) 도 함께 기록한다.
"""
import json, sys, glob, os
import torch
src, dst = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(src, "expert_distribution_recorder_*.pt")))
assert files, f"no recorder dump in {src}"
d = torch.load(files[-1], map_location="cpu", weights_only=False)
lc = d["logical_count"]
if lc.dim() == 3:              # (steps, layers, experts) 일 수도 있음 → 합산
    lc = lc.sum(0)
lc = lc.to(torch.float64)
L, E = lc.shape
order = torch.argsort(lc, dim=1, descending=True)          # [L, E] logical ids by freq
p2l = order.tolist()
json.dump({"physical_to_logical_map": p2l}, open(dst, "w"))
tot = lc.sum(1, keepdim=True).clamp(min=1)
srt = torch.sort(lc, dim=1, descending=True).values / tot
cov = {n: (srt[:, :n].sum(1)) for n in (16, 40, 64, 80, 96, 112, 128)}
stats = {"source": os.path.basename(files[-1]), "layers": L, "experts": E,
         "total_calls": int(lc.sum().item()),
         "coverage_mean": {str(n): float(c.mean()) for n, c in cov.items()},
         "coverage_min": {str(n): float(c.min()) for n, c in cov.items()},
         "coverage_max": {str(n): float(c.max()) for n, c in cov.items()}}
json.dump(stats, open(dst.replace(".json", "_stats.json"), "w"), indent=1)
print(f"layers={L} experts={E} calls={stats['total_calls']:,} → {dst}")
for n in (64, 80, 96, 112, 128):
    print(f"  top-{n:3d} coverage mean {100*stats['coverage_mean'][str(n)]:.1f}%  min {100*stats['coverage_min'][str(n)]:.1f}%")
