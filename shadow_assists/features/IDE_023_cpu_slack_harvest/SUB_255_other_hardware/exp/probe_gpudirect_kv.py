"""사용자 축: CPU/DRAM 미리계산 KV → GPUDirect 주입 → GPU prefill 재연산 skip.
feasibility: 재사용 KV를 GPUDirect-급(pinned H2D / peer DMA) 대역폭으로 주입하는 시간이
GPU prefill 재연산 시간보다 작은가? 작으면 공유-프리픽스 워크로드서 win.
KV 크기(70B): layers80 × 2(K,V) × kv_heads8(GQA) × head_dim128 × dtype.
"""
import torch, os, time
os.environ["HF_HOME"]="/raid/hf_cache"
dev="cuda:0"
# 70B KV per token
L,KVH,HD=80,8,128
for kvdt,kvb in [("fp16",2),("fp8",1)]:
    per_tok=L*2*KVH*HD*kvb
    print(f"\n=== KV dtype={kvdt}: {per_tok/1024:.0f} KB/token ===", flush=True)
    for P in [2000, 8000]:
        nbytes=per_tok*P
        # pinned DRAM→GPU H2D 대역폭 (GPUDirect-급 경로)
        src=torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        dst=torch.empty(nbytes, dtype=torch.uint8, device=dev)
        torch.cuda.synchronize()
        for _ in range(3): dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        N=10; t0=time.perf_counter()
        for _ in range(N): dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)/N
        bw=nbytes/dt/1e9
        inj_ms=dt*1000
        print(f"  P={P:5d} 토큰: KV {nbytes/1e6:6.0f} MB | 주입 {inj_ms:6.2f} ms @ {bw:5.0f} GB/s", flush=True)
        del src,dst; torch.cuda.empty_cache()
print("\n참고: 70B FP4 prefill TTFT(P=2000) ≈ 수십~수백 ms (GPU 측정 필요). 주입 시간이 그보다",flush=True)
print("작으면 캐시 적중 시 prefill 회피 win. P 클수록(긴 공유프리픽스) 이득↑.", flush=True)
print("(pin H2D는 GPUDirect-Storage[NVMe직접]·peer DMA 상한의 보수적 프록시)", flush=True)
