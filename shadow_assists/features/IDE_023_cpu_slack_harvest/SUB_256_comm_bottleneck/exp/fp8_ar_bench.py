"""iter4: FP8 저정밀 multimem all_reduce vs bf16 (decode 크기). 통신량 절반 → latency↓?
torchrun --nproc_per_node=8 fp8_ar_bench.py
multimem이 fp8(e4m3/e5m2) 리덕션 지원하는지 + 속도. 지원 안하면 fp8 AR 경로 死.
"""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
torch.cuda.set_device(rank); dev=f"cuda:{rank}"
import torch.distributed._symmetric_memory as symm_mem
grpname=dist.group.WORLD.group_name
def bench(fn,x,N=50):
    for _ in range(10): fn(x)
    torch.cuda.synchronize(); dist.barrier()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
    for _ in range(N): fn(x)
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en)/N*1000
def mm(t): return torch.ops.symm_mem.multimem_all_reduce_(t,"sum",grpname)
if rank==0: print(f"\n##### FP8 vs bf16 multimem AR (decode), world={world} #####",flush=True)
for smb in [1.0, 2.75]:
    for dt,name,b in [(torch.bfloat16,"bf16",2),(torch.float16,"fp16",2),(torch.float8_e4m3fn,"fp8e4m3",1),(torch.float8_e5m2,"fp8e5m2",1)]:
        n=int(smb*1024*1024/b)
        try:
            xs=symm_mem.empty(n,dtype=dt,device=dev); xs.fill_(1.0 if 'fp8' not in name else 0.0)
            symm_mem.rendezvous(xs,grpname)
            lat=bench(mm,xs)
            if rank==0: print(f"  {smb:.2f}MB {name:8s}: {lat:7.2f} us",flush=True)
        except Exception as e:
            if rank==0: print(f"  {smb:.2f}MB {name:8s}: 미지원 ({str(e)[:60]})",flush=True)
dist.barrier(); dist.destroy_process_group()
