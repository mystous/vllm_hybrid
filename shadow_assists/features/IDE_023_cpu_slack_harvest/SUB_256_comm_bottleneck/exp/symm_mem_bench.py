"""iter2: symm_mem(multimem-PTX) one-shot/two-shot all_reduce vs NCCL 직접 비교 (decode 크기).
torchrun --nproc_per_node=8 symm_mem_bench.py
PyTorch symm_mem 은 NVLink multimem.ld_reduce/st PTX 로 in-fabric 리덕션 → latency-bound 소형 AR 최적.
"""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank=dist.get_rank(); world=dist.get_world_size()
torch.cuda.set_device(rank); dev=f"cuda:{rank}"
GRP="0"
try:
    import torch.distributed._symmetric_memory as symm_mem
    HAVE=True
except Exception as e:
    HAVE=False
    if rank==0: print("symm_mem import 실패:", e, flush=True)

def bench(fn, x, N=50):
    for _ in range(10): fn(x)
    torch.cuda.synchronize(); dist.barrier()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
    for _ in range(N): fn(x)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/N*1000  # us

sizes_mb=[0.0625,0.25,0.5,1.0,2.75,8.0]
if rank==0:
    print(f"\n##### symm_mem(multimem-PTX) vs NCCL, world={world} #####", flush=True)
    print(f"{'size_MB':>8} {'nccl_us':>9} {'1shot_us':>9} {'2shot_us':>9} {'mmem_us':>9} {'best':>8}", flush=True)

# symm_mem group 활성
if HAVE:
    try:
        symm_mem.enable_symm_mem_for_group(GRP if dist.group.WORLD is None else dist.group.WORLD.group_name)
    except Exception:
        pass

for smb in sizes_mb:
    n=int(smb*1024*1024/2)
    # NCCL
    xn=torch.ones(n, dtype=torch.bfloat16, device=dev)
    lat_nccl=bench(lambda t: dist.all_reduce(t), xn)
    lat1=lat2=float('nan')
    if HAVE:
        try:
            grpname=dist.group.WORLD.group_name
            xs=symm_mem.empty(n, dtype=torch.bfloat16, device=dev); xs.fill_(1.0)
            symm_mem.rendezvous(xs, grpname)
            def one(t): return torch.ops.symm_mem.one_shot_all_reduce(t, "sum", grpname)
            def two(t): return torch.ops.symm_mem.two_shot_all_reduce_(t, "sum", grpname)
            def mm(t): return torch.ops.symm_mem.multimem_all_reduce_(t, "sum", grpname)
            lat1=bench(one, xs)
            lat2=bench(two, xs)
            try:
                lat_mm=bench(mm, xs)
            except Exception as e:
                lat_mm=float('nan')
                if rank==0 and smb==sizes_mb[0]: print("  multimem op 실패:", str(e)[:100], flush=True)
        except Exception as e:
            if rank==0 and smb==sizes_mb[0]: print("  symm_mem op 실패:", str(e)[:120], flush=True)
    if rank==0:
        cands={"nccl":lat_nccl}
        if lat1==lat1: cands["1shot"]=lat1
        if lat2==lat2: cands["2shot"]=lat2
        try:
            if lat_mm==lat_mm: cands["mmem"]=lat_mm
        except NameError: lat_mm=float('nan')
        best=min(cands, key=cands.get)
        print(f"{smb:8.3f} {lat_nccl:9.2f} {lat1:9.2f} {lat2:9.2f} {lat_mm:9.2f} {best:>8}", flush=True)
dist.barrier(); dist.destroy_process_group()
