#!/usr/bin/env python3
"""IDE_053 TriX-on-PCIe: GPU 가 호스트 메모리의 cold expert (INT4, TP 4분할) 를 직접 읽어 계산하는 경로의 비용.
경로 A: pinned H2D 복사 → GPU 언팩(INT4→bf16) → matmul.   경로 B: zero-copy (mapped pinned, GPU 커널이 PCIe 로 직접 읽음) 언팩 → matmul.
4 GPU 가 각자 shard 를 동시에 당김 (expert 1개 = 4 shard 병렬). 출력: m 별 expert당 µs (wall, 4 GPU 병렬), 실효 GB/s.
usage: trix_pcie_bench.py <out.json> [mode=all|pull]   (pull = 간섭 실험용 무한 PCIe 읽기 루프, 60s)"""
import sys, time, json, threading, os
import numpy as np, torch, cupy as cp
H, I, E_TOTAL = 6144, 2560, 160
NG = 4
BYTES_EXPERT = (H * I * 3) // 2              # INT4: gate, up, down = 23.6 MB
SHARD = BYTES_EXPERT // NG                   # per-GPU shard 5.9 MB
NEXP = 64                                    # 호스트 풀에 둘 expert 수 (1.5 GB, L3 초과)
out_f = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "all"
CPU_REF = {1: 66, 2: 72, 4: 71, 8: 120, 16: 212, 32: 495, 64: 1181*0.85, 128: 275, 256: 500}  # kt AMX/AVX 실측 (µs/expert, pool)

# ---- 호스트 풀: mapped pinned (zero-copy 가능) ----
total = NEXP * BYTES_EXPERT
hptr = cp.cuda.runtime.hostAlloc(total, 1 | 2)   # Portable | Mapped
import ctypes
buf = (ctypes.c_uint8 * total).from_address(hptr)
hnp = np.ctypeslib.as_array(buf)
hnp[:] = np.random.randint(0, 256, size=total, dtype=np.uint8)

def shard_ptr(e, g): return hptr + e * BYTES_EXPERT + g * SHARD

results = []
def run_gpu(g, m, n_exp, zero_copy, times):
    cp.cuda.Device(g).use(); torch.cuda.set_device(g)
    x = torch.randn(m, H, dtype=torch.bfloat16, device=f"cuda:{g}")
    # shard 를 gate/up (H × I/4 × 2) 와 down (I/4 × H) 로 해석: 언팩 후 bf16 행렬
    n_gu = H * (I // NG) * 2; n_dn = (I // NG) * H
    dev_u8 = torch.empty(SHARD, dtype=torch.uint8, device=f"cuda:{g}")
    stream = torch.cuda.Stream(device=g)
    with torch.cuda.stream(stream):
        for it in range(n_exp + 3):
            e = it % NEXP
            if it == 3: torch.cuda.synchronize(g); t0 = time.perf_counter()
            if zero_copy:
                dptr = cp.cuda.runtime.hostGetDevicePointer(shard_ptr(e, g), 0)
                mem = cp.cuda.UnownedMemory(dptr, SHARD, None); mp = cp.cuda.MemoryPointer(mem, 0)
                src = cp.ndarray((SHARD,), dtype=cp.uint8, memptr=mp)
                dst = cp.asarray(src)          # GPU 커널이 mapped 호스트 메모리를 PCIe 로 직접 읽어 device 로 (zero-copy read)
                u8 = torch.as_tensor(dst, device=f"cuda:{g}")
            else:
                src = np.ctypeslib.as_array((ctypes.c_uint8 * SHARD).from_address(shard_ptr(e, g)))
                dev_u8.copy_(torch.from_numpy(src), non_blocking=True)   # pinned → device (copy engine)
                u8 = dev_u8
            lo = (u8 & 0x0F).to(torch.int8); hi = (u8 >> 4).to(torch.int8)
            w = torch.cat([lo, hi]).to(torch.bfloat16)                    # 2×SHARD 원소 = 전체 nibble
            w_gu = w[:n_gu].view(H, (I // NG) * 2); w_dn = w[n_gu:n_gu + n_dn].view(I // NG, H)
            h = x @ w_gu; a = h[:, :I // NG] * h[:, I // NG:]; y = a @ w_dn
        torch.cuda.synchronize(g); dt = (time.perf_counter() - t0) / n_exp
    times[g] = dt

if mode == "pull":
    # 간섭 실험: 4 GPU 가 60초 동안 계속 호스트 메모리를 당김 (m=1, 복사 경로)
    stop = time.time() + 60
    def loop(g):
        while time.time() < stop:
            t = {}; run_gpu(g, 1, 64, False, t)
    th = [threading.Thread(target=loop, args=(g,)) for g in range(NG)]
    [t.start() for t in th]; [t.join() for t in th]; print("pull done"); raise SystemExit

for zero_copy in (False, True):
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        n_exp = 64 if m <= 128 else 32
        times = {}
        th = [threading.Thread(target=run_gpu, args=(g, m, n_exp, zero_copy, times)) for g in range(NG)]
        [t.start() for t in th]; [t.join() for t in th]
        wall = max(times.values()) * 1e6          # expert 1개 = 4 shard 병렬 → 가장 느린 GPU
        gbs = BYTES_EXPERT / (wall * 1e-6) / 1e9  # expert 바이트 / wall
        per_gpu_gbs = SHARD / (wall * 1e-6) / 1e9
        rec = dict(path="zero_copy" if zero_copy else "h2d_copy", m=m, us_per_expert=round(wall, 1), eff_GBps_total=round(gbs, 1),
                   per_gpu_GBps=round(per_gpu_gbs, 1), cpu_ref_us=CPU_REF.get(m))
        print("path=%-9s m=%5d  GPU us/expert=%8.1f (per-GPU %5.1f GB/s, expert 집계 %6.1f GB/s)   CPU AMX/AVX ref=%s" % (
            rec["path"], m, wall, per_gpu_gbs, gbs, rec["cpu_ref_us"]), flush=True)
        results.append(rec)
json.dump(results, open(out_f, "w"), indent=1); print("saved", out_f)
