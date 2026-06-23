"""각도4 (수정): gate 출력을 top-1로 절단해 진짜 top-1 draft vs top-2 full accept 측정."""
import torch, os, numpy as np
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="mistralai/Mixtral-8x7B-Instruct-v0.1"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
blocks=[m for m in model.modules() if m.__class__.__name__=="MixtralSparseMoeBlock"]
DK=[2]  # draft top_k
for b in blocks:
    g=b.gate
    of=g.forward
    def mk(of):
        def f(h):
            out=of(h)  # (router_logits, top_k_weights, top_k_index) 또는 유사
            if DK[0]>=2: return out
            rl, w, idx = out
            w1=w[..., :1]; w1=w1/w1.sum(-1,keepdim=True)  # top-1 재정규화
            return rl, w1, idx[..., :1]
        return f
    g.forward=mk(of)
texts=["The capital of France is Paris and the largest planet in the solar system is Jupiter which has",
 "def quicksort(arr):\n    if len(arr)<=1: return arr\n    pivot=arr[len(arr)//2]\n    return quicksort([x for x in arr if x<pivot])+[pivot]",
 "Machine learning minimizes a loss via gradient descent and backpropagation through the computational graph to update",
 "한국의 수도는 서울이며 한강이 도시를 가로질러 흐르고 경제와 문화의 중심지 역할을 하며 인구는 약",
 "In quantum mechanics the wave function collapses upon measurement and the uncertainty principle limits",
 "The HTTP protocol uses a request-response model where clients send requests and servers return responses with status",
 "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by",
 "To solve a quadratic equation we use the formula negative b plus or minus square root of b squared minus"]
@torch.no_grad()
def argmax_of(t,dk):
    DK[0]=dk; ids=tok(t,return_tensors="pt").input_ids.to(dev); return model(ids).logits[0].float().argmax(-1)
accs=[]
print("MoE-spec 진짜 accept (top-1 draft vs top-2 full)")
for t in texts:
    ref=argmax_of(t,2); d1=argmax_of(t,1)
    a=(d1==ref).float().mean().item(); accs.append(a)
print(f"  per-text accept: {[round(x,3) for x in accs]}")
m=np.mean(accs); c=0.5
print(f"  평균 accept={m:.3f}, draft연산비 c={c} → 속도배율 1/(c+(1-a))={1/(c+(1-m)):.2f}")
print("판정: >1.1 유망. (단 expert연산은 전체의 ~20%라 전체속도는 이보다 작음)")
