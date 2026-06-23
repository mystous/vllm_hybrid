"""각도4: MoE speculative — 저비용 draft(top-1 expert)가 full(top-2) 출력과 일치하는 천장.
높으면: top-1 draft로 토큰 제안 + top-2 verify(rejection sampling=출력동등) → 평균 expert 연산↓.
vLLM엔 reduced-expert self-draft 없음=novel. Mixtral-8x7B(8expert top-2)로 cheap 측정.
"""
import torch, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="mistralai/Mixtral-8x7B-Instruct-v0.1"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
# MoE 블록 top-k 동적 패치
TOPK=[2]
import torch.nn.functional as F
blocks=[]
for mod in model.modules():
    if mod.__class__.__name__=="MixtralSparseMoeBlock": blocks.append(mod)
print(f"MoE blocks={len(blocks)}, experts_per_tok(full)={blocks[0].top_k if blocks else '?'}")
orig_topk=blocks[0].top_k if blocks else 2
def patch(b):
    of=b.forward
    def f(hidden):
        b.top_k=TOPK[0]
        return of(hidden)
    return f
for b in blocks: b.forward=patch(b)
texts=[
 "The capital of France is Paris and the largest planet in the solar system is Jupiter which has a mass greater than all other planets combined and its great red spot is a storm",
 "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
 "Machine learning models are trained by minimizing a loss function using gradient descent which iteratively updates parameters in the direction of steepest descent computed via backpropagation through the computational graph",
 "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation in the electron transport chain located in the inner membrane",
 "In quantum mechanics, the wave function collapses upon measurement, and the Heisenberg uncertainty principle states that position and momentum cannot both be known precisely",
 "한국의 수도는 서울이며 인구는 약 천만명이고 한강이 도시를 가로질러 흐르며 경제와 문화의 중심지 역할을 한다",
 "To solve the quadratic equation ax^2 + bx + c = 0, we use the formula x equals negative b plus or minus the square root of b squared minus four a c all over two a",
 "The stock market experienced significant volatility today as investors reacted to the central bank's announcement regarding interest rate policy and inflation expectations for the coming quarter",
 "Once upon a time in a distant kingdom there lived a young princess who dreamed of exploring the world beyond the castle walls despite her parents' wishes",
 "The HTTP protocol operates on a request-response model where clients send requests to servers which process them and return responses containing status codes headers and body content",
 "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll in the chloroplasts of plant cells",
 "Recursion is a programming technique where a function calls itself to solve smaller subproblems until reaching a base case that terminates the recursive descent",
]
@torch.no_grad()
def lg(t):
    ids=tok(t,return_tensors="pt").input_ids.to(dev); return model(ids).logits[0].float()
print("MoE-spec draft accept 천장 (top-1 draft vs top-2 full)")
accs=[]
for t in texts:
    TOPK[0]=orig_topk; ref=lg(t).argmax(-1)
    TOPK[0]=1; d1=lg(t).argmax(-1)
    a=(d1==ref).float().mean().item(); accs.append(a)
    print(f"  top1-draft accept={a:.3f}")
import numpy as np
mean=np.mean(accs)
# self-spec 경제성: draft 비용 c≈top1/top2=1/2 expert. 배율 1/(c+(1-a))
c=1/orig_topk
print(f"\n평균 accept={mean:.3f}, draft 연산비 c={c:.2f} → 속도배율 1/(c+(1-a))={1/(c+(1-mean)):.2f}")
print("판정: 배율>1.1이면 MoE-spec 유망(구현가치). <1이면 死(R5 layer-skip식 coupling).")
