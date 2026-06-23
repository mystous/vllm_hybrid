"""각도1: Persistent FP8 residual stream — 레이어 간 활성을 FP8로 유지(재양자화 제거).
vLLM은 매 레이어 bf16 residual → FP8 quant(per_token_group, nsys 13%). residual을 FP8로 유지하면
그 변환 제거 가능. 단 61층 누적 정확도가 관건. 각 레이어 출력(residual)을 FP8로 라운딩하고
출력 분포 유지되는지 측정. 게이트 통과면 신규 win(quant 오버헤드 제거).
"""
import torch, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
m=model.model; nl=len(m.layers)
def to_fp8(x, group=128):  # per-group(128) 동적 스케일 e4m3
    o=x.shape; xf=x.reshape(-1,group); s=xf.abs().amax(-1,keepdim=True).clamp_min(1e-6)/448
    q=(xf/s).clamp(-448,448).to(torch.float8_e4m3fn).to(x.dtype)*s
    return q.reshape(o)
texts=["The capital of France is Paris and the largest planet in the solar system is",
       "def fib(n):\n    return n if n<2 else fib(n-1)+fib(n-2)\nprint(fib(10))",
       "In machine learning, gradient descent iteratively minimizes a loss by"]
import types
FP8_RESID=[False]
# 각 decoder layer 출력(hidden=residual)을 FP8로 라운딩하는 hook
orig_fwds={}
def wrap(layer):
    of=layer.forward
    def f(*a, **k):
        out=of(*a, **k)
        if FP8_RESID[0]:
            hs=out[0] if isinstance(out,tuple) else out
            hs=to_fp8(hs)
            out=(hs,)+out[1:] if isinstance(out,tuple) else hs
        return out
    return f
for L in m.layers: orig_fwds[L]=L.forward; L.forward=wrap(L)
@torch.no_grad()
def logits(t):
    ids=tok(t,return_tensors="pt").input_ids.to(dev)
    return model(ids).logits[0].float()
print("Persistent FP8 residual 정확도 (8B, 32층)")
for t in texts:
    FP8_RESID[0]=False; ref=logits(t); ra=ref.argmax(-1)
    FP8_RESID[0]=True;  fp=logits(t); fa=fp.argmax(-1)
    top1=(fa==ra).float().mean().item()
    md=(fp-ref).abs().max().item()
    import math
    def ppl(lg,ids):
        lp=torch.log_softmax(lg,-1); return math.exp(-lp[torch.arange(len(ids)),ids].mean().item())
    print(f"  top1={top1:.3f}  max_logit_diff={md:.2f}")
print("\n판정: top1≈1.0 & logit_diff 작으면(<~2) FP8 residual viable → 재양자화 13% 제거 신규 win.")
print("(FP8 residual이 게이트 통과하면 대형 FP8모델서 quant 커널 오버헤드 제거 가능)")
