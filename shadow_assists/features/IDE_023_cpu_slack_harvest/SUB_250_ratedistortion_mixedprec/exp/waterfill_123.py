"""SUB_250 1~3단계: D_i(b) 곡선 → water-filling 배분 → uniform-4bit와 joint 왜곡 비교.
왜곡=teacher-forcing 단일 forward의 mean-KL(baseline‖perturbed). b∈{2,4,8} per-group(16) uniform."""
import torch, os, math, collections
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map=dev).eval()
def quant(x,b,group=16):  # per-group(16) 대칭 uniform b-bit
    if b>=16: return x
    o=x.shape; xf=x.float().reshape(-1); n=xf.numel(); pad=(group-n%group)%group
    if pad: xf=torch.nn.functional.pad(xf,(0,pad))
    g=xf.reshape(-1,group); qmax=2**(b-1)-1; s=g.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qmax
    dq=(torch.round(g/s).clamp(-qmax,qmax)*s).reshape(-1)[:n]
    return dq.reshape(o).to(x.dtype)
text=("The data streaming accelerator processes descriptors through work queues while the memory "
      "controller arbitrates bandwidth across NUMA domains and the sampler validates tokens. ")*60
ids=tok(text,return_tensors="pt").input_ids[:,:1200].to(dev)
@torch.no_grad()
def logits(): return model(ids).logits[0].float()  # [seq,vocab]
base=torch.log_softmax(logits(),-1)
def kl(lp): return (base.exp()*(base-lp)).sum(-1).mean().item()  # mean KL over positions
layers=model.model.layers; nl=len(layers); BITS=[2,4,8]
def set_layer(li,b):
    saved={}
    for nm,m in layers[li].named_modules():
        if isinstance(m,torch.nn.Linear): saved[(li,nm)]=m.weight.data.clone(); m.weight.data=quant(m.weight.data,b)
    return saved
def restore(saved):
    for (li,nm),w in saved.items():
        dict(layers[li].named_modules())[nm].weight.data=w
# === 1단계: D_i(b) ===
print("1단계: D_i(b) 측정...")
D={}
for li in range(nl):
    for b in BITS:
        s=set_layer(li,b); D[(li,b)]=kl(torch.log_softmax(logits(),-1)); restore(s)
# === 2단계: water-filling (greedy, B_avg=4) ===
# 모든 레이어 4bit 시작 → 둔감 레이어를 2bit로 내려 절약, 그 비트로 민감 레이어 8bit 올림
# greedy: 단위비트당 왜곡변화로 4→2 강등 후보·4→8 승격 후보 매칭
import itertools
alloc={li:4 for li in range(nl)}
# 4→2 강등 비용(왜곡증가/비트절약2), 4→8 승격 이득(왜곡감소/비트추가4)
demote=sorted(range(nl), key=lambda li:(D[(li,2)]-D[(li,4)]))   # 강등시 왜곡증가 작은 순
promote=sorted(range(nl), key=lambda li:(D[(li,4)]-D[(li,8)]), reverse=True)  # 승격시 왜곡감소 큰 순
# 평균 4 유지: k개 강등(-2씩) ↔ k/2개 승격(+4씩). k 강등 → 2k 비트절약 → k/2 승격
for k in range(2, nl, 2):
    dem=demote[:k]; pro=[p for p in promote if p not in dem][:k//2]
    if len(pro)<k//2: break
    for li in dem: alloc[li]=2
    for li in pro: alloc[li]=8
    avg=sum(alloc.values())/nl
    if abs(avg-4.0)<0.3: break
    alloc={li:4 for li in range(nl)}  # 리셋 후 다음 k
avg=sum(alloc.values())/nl
print(f"2단계: water-filling 배분 (avg={avg:.2f}bit) — 2bit:{sum(v==2 for v in alloc.values())} 4bit:{sum(v==4 for v in alloc.values())} 8bit:{sum(v==8 for v in alloc.values())}")
# === 3단계: joint 검증 — mixed vs uniform-4bit (전 레이어 동시) ===
def joint(bitmap):
    saved=[]
    for li in range(nl): saved.append(set_layer(li,bitmap[li] if isinstance(bitmap,dict) else bitmap))
    d=kl(torch.log_softmax(logits(),-1))
    for s in saved: restore(s)
    return d
d_uniform4=joint(4); d_mixed=joint(alloc)
print(f"\n=== 3단계 joint 왜곡 (mean-KL, 낮을수록 좋음, 둘 다 avg≈4bit) ===")
print(f"  uniform 4bit : KL={d_uniform4:.5f}")
print(f"  water-filling: KL={d_mixed:.5f}  ({(d_mixed/d_uniform4-1)*100:+.1f}% vs uniform)")
print(f"  → mixed < uniform 이면 같은 평균비트(속도)서 정확도↑ = 이론 성립.")
