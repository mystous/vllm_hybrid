"""R6 probe: contextual activation sparsity (Deja Vu류) accept/정확도 천장.
Llama MLP: down_proj( SiLU(gate_proj(x)) * up_proj(x) ). 중간활성(4d)에서 |값| 하위 k%를 0으로
하면 그만큼 down_proj 곱셈 FLOP 절감(0 곱셈 skip). 측정: 전 레이어 MLP에 적용 시 출력 분포가
게이트(top1 일치, max logit diff) 얼마나 변하나. 거의-무손실 sparsity가 크면 GPU-direct FLOP win.
신규성: vLLM은 dense 모델 contextual activation sparsity 미적용. GPU-direct.
"""
import torch, os, types
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
m=model.model; nl=len(m.layers)
texts=[
 "The capital of France is Paris, and the capital of Japan is Tokyo. The largest planet in the solar system is",
 "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(10))\n",
 "In machine learning, gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss by",
 "Once upon a time, in a small village nestled between two mountains, there lived a young blacksmith who dreamed of",
]
PRUNE_FRAC=[0.0]  # set per run
def patched_mlp_forward(self, x):
    g=self.act_fn(self.gate_proj(x))*self.up_proj(x)   # [.., 4d] 중간활성
    fr=PRUNE_FRAC[0]
    if fr>0:
        k=int(g.shape[-1]*fr)
        if k>0:
            thr=g.abs().kthvalue(k, dim=-1, keepdim=True).values  # 하위 k번째 |값|
            g=torch.where(g.abs()<=thr, torch.zeros_like(g), g)    # 하위 k% → 0
    return self.down_proj(g)
# 패치 적용
for layer in m.layers:
    layer.mlp.forward=types.MethodType(patched_mlp_forward, layer.mlp)

@torch.no_grad()
def run(ids):
    return model(ids).logits[0]

import numpy as np
fulls={}
for i,t in enumerate(texts):
    ids=tok(t,return_tensors="pt").input_ids.to(dev)
    PRUNE_FRAC[0]=0.0
    fulls[i]=(ids, run(ids))
print("prune% | top1일치 | max_logit_diff | (MLP down_proj FLOP 절감 ≈ prune%)", flush=True)
for fr in [0.3,0.5,0.7,0.9]:
    PRUNE_FRAC[0]=fr
    t1s=[]; mds=[]
    for i,t in enumerate(texts):
        ids,fl=fulls[i]
        lg=run(ids)
        t1=(lg.argmax(-1)==fl.argmax(-1)).float().mean().item()
        md=(lg.float()-fl.float()).abs().max().item()
        t1s.append(t1); mds.append(md)
    print(f"  {int(fr*100):2d}%   | {np.mean(t1s):.3f}      | {np.mean(mds):.2f}", flush=True)
print("\n판정: 큰 prune%에서도 top1≈1.0 & logit_diff 작으면 거의-무손실 sparsity → GPU FLOP win 가능.", flush=True)
print("(주의: 이건 무손실성 상한; 실 GPU 이득엔 sparse gather 커널 필요, dense TC 대비 비효율 가능)", flush=True)
