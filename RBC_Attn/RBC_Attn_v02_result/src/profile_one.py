#!/usr/bin/env python3
"""ncu 계측용 — 정책 하나의 커널을 정확히 한 번 실행한다."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rbc import data, kernels, planner, runtime

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--pattern", default="common_private")
ap.add_argument("--nq", type=int, default=512); ap.add_argument("--nk", type=int, default=65536)
ap.add_argument("--g", type=int, default=8); ap.add_argument("--d", type=int, default=128)
ap.add_argument("--sel", type=int, default=16); ap.add_argument("--max-m", type=int, default=32)
ap.add_argument("--min-tasks", type=int, default=264)
a = ap.parse_args()

s = data.make_synthetic(a.pattern, hkv=1, nq=a.nq, nk=a.nk, g=a.g, d=a.d, bk=128, sel=a.sel, seed=0)
p = planner.plan_head(s["ptr"][0], s["block_ids"][0], bk=128, d=a.d, g=a.g,
                      max_m=a.max_m, policy=a.policy, min_tasks=a.min_tasks)
dp = runtime.upload_plan(p)
q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
kw = dict(bk=128, causal=bool(s["causal"]), scale=float(s["scale"]))
for _ in range(3):                      # JIT warmup (계측 대상 밖)
    kernels.run_plan(dp, q, k, v, qpos, **kw)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
kernels.run_plan(dp, q, k, v, qpos, **kw)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
st = p.stats(n_edges=len(s["block_ids"][0]))
print(f"POLICY={a.policy} tasks={st['n_tasks']} partial={st['partial_slots']} "
      f"padded_ratio={st.get('padded_ratio')} slots={int(p.task_qptr[-1])}")
