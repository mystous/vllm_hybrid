"""직접 통신 측정: 8-GPU all_reduce 격리 마이크로벤치 (decode AR 크기 대역).
torchrun --nproc_per_node=8 all_reduce_bench.py
NCCL_ALGO/NCCL_NVLS_ENABLE 환경으로 Ring/Tree/NVLS 비교. CUDA event 정밀 측정.
70B decode AR ≈ 168토큰×hidden8192×2B = 2.75MB. busbw = algbw×2(n-1)/n.
"""
import os, torch, torch.distributed as dist, time
dist.init_process_group("nccl")
rank=dist.get_rank(); world=dist.get_world_size()
torch.cuda.set_device(rank)
dev=f"cuda:{rank}"
algo=os.environ.get("NCCL_ALGO","default"); nvls=os.environ.get("NCCL_NVLS_ENABLE","0")
# decode AR 메시지 크기 (bytes): 70B hidden8192, bf16(2B) → tokens×16KB
sizes_mb=[0.0625,0.25,0.5,1.0,2.75,8.0]
if rank==0:
    print(f"\n##### NCCL_ALGO={algo} NVLS={nvls} world={world} #####", flush=True)
    print(f"{'size_MB':>8} {'lat_us':>9} {'algbw_GB/s':>11} {'busbw_GB/s':>11}", flush=True)
for smb in sizes_mb:
    n=int(smb*1024*1024/2)  # bf16 elem count
    x=torch.ones(n, dtype=torch.bfloat16, device=dev)
    for _ in range(10): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    N=50; st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(N): dist.all_reduce(x)
    en.record(); torch.cuda.synchronize()
    lat=st.elapsed_time(en)/N*1000  # us per all_reduce
    nbytes=n*2
    algbw=nbytes/(lat/1e6)/1e9
    busbw=algbw*2*(world-1)/world
    if rank==0:
        print(f"{smb:8.3f} {lat:9.2f} {algbw:11.1f} {busbw:11.1f}", flush=True)
    del x
dist.barrier(); dist.destroy_process_group()
