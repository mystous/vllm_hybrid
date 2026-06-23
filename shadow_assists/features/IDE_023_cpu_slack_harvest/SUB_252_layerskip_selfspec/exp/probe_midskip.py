"""R5 변종: 중간 블록 skip self-draft accept 천장.
통설=중간 레이어가 redundant, 끝 레이어는 중요. 꼬리절단(probe_layerskip) accept 낮았으니
중간 [s:e) 블록을 실제로 건너뛴 부분 forward(embed→layers[0:s]→layers[e:]→norm→lm_head)로
다음토큰 argmax가 full과 일치하는 비율 측정. 높으면 self-spec 학습없이 viable.
"""
import torch, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
m=model.model; nl=len(m.layers)
print(f"layers={nl}", flush=True)

texts=[
 "The capital of France is Paris, and the capital of Japan is Tokyo. The largest planet in the solar system is",
 "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(10))\n",
 "In machine learning, gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss by",
 "Once upon a time, in a small village nestled between two mountains, there lived a young blacksmith who dreamed of",
]
@torch.no_grad()
def full_argmax(ids):
    return model(ids).logits[0].argmax(-1)

import torch.nn as nn
_orig_layers=m.layers
@torch.no_grad()
def midskip_argmax(ids, s, e):
    # 중간 [s:e) 생략한 layer 부분집합으로 교체 후 모델 정상 forward (rotary 자체 처리)
    kept=[ _orig_layers[i] for i in (list(range(0,s))+list(range(e,nl))) ]
    m.layers=nn.ModuleList(kept)
    try:
        lg=model(ids, use_cache=False).logits[0]
    finally:
        m.layers=_orig_layers
    return lg.argmax(-1)

# 중간 블록 크기 N, 중앙 정렬
configs=[(N, (nl-N)//2, (nl-N)//2+N) for N in [2,4,8,12,16]]
agg={}
for t in texts:
    ids=tok(t,return_tensors="pt").input_ids.to(dev)
    fa=full_argmax(ids)
    for N,s,e in configs:
        arg=midskip_argmax(ids,s,e)
        match=(arg==fa).float().mean().item()
        agg.setdefault((N,s,e),[]).append(match)
print("midskip(중간 [s:e) 생략) | top1일치 | draft연산비율", flush=True)
best=None
for (N,s,e),v in agg.items():
    mr=sum(v)/len(v); comp=(nl-N)/nl
    print(f"  skip[{s}:{e}] (N={N:2d}) | top1={mr:.3f} | draft연산={comp:.2f}", flush=True)
    if best is None or mr>best[1]: best=((N,s,e),mr)
print(f"\n꼬리절단 대비: skip2 꼬리=0.537 / skip8 꼬리=0.354 (probe_layerskip)", flush=True)
print(f"최고 midskip: {best}", flush=True)
print("판정: 같은 N에서 midskip top1 ≫ 꼬리절단이고 >0.6이면 self-spec 학습없이 viable→GO. 아니면 R5 기각.", flush=True)
