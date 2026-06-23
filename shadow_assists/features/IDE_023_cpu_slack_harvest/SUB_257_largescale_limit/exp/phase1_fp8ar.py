"""Phase 1: FP8 all-reduce 실제 구현 — 통신량 2×↓ 가설 코드검증.
torchrun --nproc_per_node=8 phase1_fp8ar.py
구현: all_to_all(FP8 전송, per-channel scale) + 로컬 fp32 누적 + all_gather(bf16).
비교: torch bf16 all_reduce(현행). 측정: 지연(us) + 합산 정밀도(rel_err vs 참 bf16 합).
"""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); rank=dist.get_rank(); W=dist.get_world_size()
torch.cuda.set_device(rank); dev=f"cuda:{rank}"

def bench(fn, *a, N=50):
    for _ in range(8): fn(*a)
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(N): fn(*a)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/N*1000

def bf16_ar(x):
    y=x.clone(); dist.all_reduce(y); return y

def fp8_ar(x):
    # x: [N,H] bf16. 행을 W조각으로 나눠 all_to_all(FP8) → 각 rank가 자기 shard의 W개 기여를 fp32 합 → all_gather.
    N,H=x.shape; assert N%W==0; sh=N//W
    # per-(channel) scale: 열 단위 amax
    scale=x.abs().amax(0,keepdim=True).clamp_min(1e-6)/448.0   # [1,H]
    xq=(x/scale).clamp(-448,448).to(torch.float8_e4m3fn)        # FP8 전송본
    # all_to_all: rank i의 j번째 청크 → rank j
    send=list(xq.view(W,sh,H).unbind(0))
    recv=[torch.empty(sh,H,dtype=torch.float8_e4m3fn,device=dev) for _ in range(W)]
    dist.all_to_all(recv, send)                                  # FP8 전송(통신량 1/2 of bf16)
    # 각 rank: 자기 shard에 모인 W개 FP8 기여를 fp32로 합 (scale은 all_gather 필요 → 근사: 평균 scale 브로드캐스트 생략, 송신 scale 동봉 필요)
    # 정확 합엔 각 송신자의 scale 필요 → scale도 all_to_all
    sc_send=list(scale.expand(W,H).contiguous().unbind(0)) if False else None
    # 간단화: 전 rank scale을 all_gather (작음)
    allsc=[torch.empty_like(scale) for _ in range(W)]
    dist.all_gather(allsc, scale)
    acc=torch.zeros(sh,H,dtype=torch.float32,device=dev)
    for j in range(W):
        acc += recv[j].to(torch.float32)*allsc[j].to(torch.float32)  # dequant with sender scale
    accb=acc.to(torch.bfloat16)
    out=torch.empty(N,H,dtype=torch.bfloat16,device=dev)
    dist.all_gather(list(out.view(W,sh,H).unbind(0)), accb)
    return out

H=16384; N=W*16   # 405B hidden=16384, 128 tokens
x=torch.randn(N,H,device=dev,dtype=torch.bfloat16)
mask=torch.rand(N,H,device=dev)<0.03; x=x+mask*torch.randn_like(x)*8  # outlier
ref=x.clone(); dist.all_reduce(ref)   # 참 bf16 합

t_bf=bench(bf16_ar, x)
t_fp=bench(fp8_ar, x)
out_fp=fp8_ar(x)
rel=(out_fp.float()-ref.float()).norm()/ref.float().norm()
maxd=(out_fp.float()-ref.float()).abs().max()
if rank==0:
    print(f"\n##### Phase1 FP8 all-reduce (H={H}, N={N}, W={W}) #####")
    print(f"  bf16 AR(현행): {t_bf:.1f} us")
    print(f"  FP8  AR(신규): {t_fp:.1f} us  → {(1-t_fp/t_bf)*100:+.0f}% 지연")
    print(f"  FP8 합산 rel_err={rel.item():.4f}  max_abs={maxd.item():.3f}")
    print(f"  통신량: bf16 2×(W-1)/W vs FP8 all_to_all(1/2) — 이론상 ~절반")
    print("  판정: 지연↓ AND rel_err 보상가능(<3%)면 Phase2(오차보상)로.")
dist.barrier(); dist.destroy_process_group()
