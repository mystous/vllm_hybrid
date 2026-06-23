"""R7 probe: 양자화 self-speculative decoding accept 천장.
draft = 같은 모델의 저비트(2/3bit) 양자화, target = 4bit(FP4 배포 구성). decode는 memory-bound라
draft 적재바이트 ∝ 비트폭 → draft 연산비율 c ≈ b_draft/4 (layer-skip의 c≈1 문제 회피).
측정: b-bit 가중치(group16) top1이 4bit target과 얼마나 일치(accept) → 1/(c+(1-a)) 속도배율.
신규: vLLM 양자화 self-draft 미적용.
"""
import torch, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
base=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
import torch.nn as nn
def quant_w(x,b,group=16):
    if b>=16: return x
    o=x.shape; xf=x.float().reshape(-1); n=xf.numel(); pad=(group-n%group)%group
    if pad: xf=torch.nn.functional.pad(xf,(0,pad))
    g=xf.reshape(-1,group); qmax=2**(b-1)-1; s=g.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qmax
    return (torch.round(g/s).clamp(-qmax,qmax)*s).reshape(-1)[:n].reshape(o).to(x.dtype)
orig={}
for nm,p in base.named_parameters():
    if p.dim()==2 and ("proj" in nm or "mlp" in nm): orig[nm]=p.data.clone()
@torch.no_grad()
def quantize_all(b):
    for nm,p in base.named_parameters():
        if nm in orig: p.data=quant_w(orig[nm],b)
@torch.no_grad()
def restore():
    for nm,p in base.named_parameters():
        if nm in orig: p.data=orig[nm]
texts=[
 "The capital of France is Paris, and the capital of Japan is Tokyo. The largest planet in the solar system is",
 "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(10))\n",
 "In machine learning, gradient descent iteratively adjusts parameters to minimize a loss by computing",
 "Once upon a time, in a small village nestled between two mountains, there lived a young blacksmith who dreamed of",
]
import numpy as np
@torch.no_grad()
def argmax_for(b, ids):
    quantize_all(b); a=base(ids).logits[0].argmax(-1); return a
# target = 4bit
print("draft_b | accept(top1 vs 4bit target) | c=b/4 | 속도배율 1/(c+(1-a))", flush=True)
tgt={}
for i,t in enumerate(texts):
    ids=tok(t,return_tensors="pt").input_ids.to(dev); tgt[i]=(ids, argmax_for(4, ids))
for b in [2,3]:
    accs=[]
    for i,t in enumerate(texts):
        ids,ta=tgt[i]; da=argmax_for(b, ids)
        accs.append((da==ta).float().mean().item())
    a=float(np.mean(accs)); c=b/4
    sp=1.0/(c+(1-a))
    print(f"   {b}bit | accept={a:.3f} | c={c:.2f} | 배율={sp:.2f}", flush=True)
restore()
print("\n판정: 배율>1.1 인 (b,accept) 있으면 양자화 self-draft GO(vLLM 통합). 전부 ≤1이면 R7 기각.", flush=True)
print("(주의: c=b/4는 memory-bound 가정 상한; verify 비용·2모델 메모리·실커널 별도 고려 필요)", flush=True)
