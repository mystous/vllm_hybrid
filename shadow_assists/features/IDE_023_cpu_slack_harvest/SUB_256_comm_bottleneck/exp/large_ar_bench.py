"""iter7: 큰 AR(prefill 크기 16~128MB)서 multimem vs NCCL-Ring vs 2-shot 격리 비교.
큰 AR을 어디로 라우팅해야 최선인지 (multimem max_size 패치 가치 판정).
torchrun --nproc_per_node=8 large_ar_bench.py
"""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
torch.cuda.set_device(rank); dev=f"cuda:{rank}"
import torch.distributed._symmetric_memory as symm_mem
grp=dist.group.WORLD.group_name
def bench(fn,x,N=30):
    for _ in range(5): fn(x)
    torch.cuda.synchronize(); dist.barrier()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
    for _ in range(N): fn(x)
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en)/N*1000
if rank==0: print(f"\n##### 큰 AR: multimem vs NCCL-Ring vs 2-shot, world={world} #####\n{'MB':>6} {'nccl_us':>9} {'2shot_us':>9} {'mmem_us':>9} {'best':>7}",flush=True)
for smb in [16,32,64,128]:
    n=int(smb*1024*1024/2)
    xn=torch.ones(n,dtype=torch.bfloat16,device=dev)
    os.environ["NCCL_ALGO"]="Ring"
    ln=bench(lambda t:dist.all_reduce(t),xn)
    try:
        xs=symm_mem.empty(n,dtype=torch.bfloat16,device=dev); xs.fill_(1.0); symm_mem.rendezvous(xs,grp)
        l2=bench(lambda t:torch.ops.symm_mem.two_shot_all_reduce_(t,"sum",grp),xs)
        lm=bench(lambda t:torch.ops.symm_mem.multimem_all_reduce_(t,"sum",grp),xs)
    except Exception as e:
        l2=lm=float('nan')
        if rank==0: print("  symm 실패:",str(e)[:80],flush=True)
    if rank==0:
        c={"nccl":ln,"2shot":l2,"mmem":lm}; best=min((k for k in c if c[k]==c[k]),key=lambda k:c[k])
        print(f"{smb:6d} {ln:9.1f} {l2:9.1f} {lm:9.1f} {best:>7}",flush=True)
    del xn
dist.barrier(); dist.destroy_process_group()
