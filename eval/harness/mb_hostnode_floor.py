#!/usr/bin/env python3
"""host 노드 디스패치 바닥 마이크로벤치: CUDA graph 안에 (tiny kernel → host 노드) × K 를 캡처해 재생.
 host 노드 = kt_kernel_ext CPUInfer.submit_with_cuda_stream (no-op 급 task) / sync_with_cuda_stream (큐 빈 상태 sync)
 memop 노드 = wait_signal_on_stream (플래그 미리 1) 로 대체 시 비용 비교."""
import torch, time, ctypes, sys
from kt_kernel import kt_kernel_ext
dev = torch.device("cuda:0")
ci = kt_kernel_ext.CPUInfer(2)
x = torch.randn(64, 6144, device=dev)
def kern(): return x.mul_(1.0001)
def run_graph(K, mode):
    st = torch.cuda.Stream(); s = st.cuda_stream
    with torch.cuda.stream(st):
        for _ in range(3): kern()
    torch.cuda.synchronize()
    if mode == "memop":
        for k in range(K): ci.write_flag_on_stream(s, 100 + k, 1)   # 사전 세트 (eager)
        torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        for k in range(K):
            kern()
            if mode == "host_sync": ci.sync_with_cuda_stream(s, 0)
            elif mode == "memop":
                ci.wait_signal_on_stream(s, 100 + k)      # wait>=1 then write 0
                ci.write_flag_on_stream(s, 100 + k, 1)    # 다음 재생을 위해 다시 1 (같은 graph 안)
            # mode == "none": 커널만
    torch.cuda.synchronize()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 20 * 1000  # us per replay
for K in (62, 186):
    base = run_graph(K, "none")
    hs = run_graph(K, "host_sync")
    mo = run_graph(K, "memop")
    print(f"K={K}: kernel-only {base:.0f}us | +host sync 노드 {hs:.0f}us → 노드당 {(hs-base)/K:.1f}us | +memop(wait+write) {mo:.0f}us → 노드당 {(mo-base)/K:.1f}us", flush=True)
