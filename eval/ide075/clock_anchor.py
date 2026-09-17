#!/usr/bin/env python3
"""IDE_075 §7.4 — CPU(CLOCK_REALTIME) ↔ kineto GPU 시계 anchor (부하 밖, 컨테이너 sgl-kt 에서 실행).
방법: torch.profiler(CPU+CUDA) 안에서 작은 커널을 launch 하고 cudaDeviceSynchronize 직후 host time.time_ns() 를 찍는다.
kineto 가 host 시계로 변환한 커널 종료 시각 t_k_end 와 host t_after_sync 의 차이 d = t_after_sync − t_k_end 는 (sync 반환 지연 ≥0) + (GPU→host 정렬 offset). d 의 최소값이 offset 의 상한(양의 방향) 이고, d 가 음수면 offset 이 음수임이 확정된다.
반복 N=200, 결과: min/p50/max d, 음수 개수. 정렬 오차의 절대 경계가 아니라 경험적 범위. 산출: JSON (stdout)."""
import json, time, torch, tempfile, os, statistics as st
from torch.profiler import profile, ProfilerActivity
N = 200; x = torch.randn(1024, 1024, device="cuda"); torch.cuda.synchronize()
hosts = []
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for i in range(N):
        y = x @ x; torch.cuda.synchronize(); hosts.append(time.time_ns())
f = tempfile.mktemp(suffix=".json"); p.export_chrome_trace(f); j = json.load(open(f)); base = j.get("baseTimeNanoseconds", 0)
ks = sorted((base + int(round(e["ts"] * 1000)), base + int(round((e["ts"] + e["dur"]) * 1000))) for e in j["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel")
ks = ks[-N:]
d = [(h - ke) / 1e3 for h, (kb, ke) in zip(hosts, ks)]   # µs
res = {"n": len(d), "d_min_us": min(d), "d_p50_us": st.median(d), "d_max_us": max(d), "negative_count": sum(1 for v in d if v < 0), "method": "host time after cudaDeviceSynchronize − kineto kernel end (host-aligned)", "interpretation": "min(d) = sync 지연 + offset 상한; 음수 = GPU 시계가 host 보다 앞섬(offset<0)", "trace_base_ns": base, "torch": torch.__version__}
print(json.dumps(res))
