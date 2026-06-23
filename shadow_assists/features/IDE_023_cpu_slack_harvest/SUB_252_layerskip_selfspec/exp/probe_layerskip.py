"""R5 타당성 probe: layer-skip self-speculative decoding accept 천장.
가설: 같은 모델을 마지막 N개 레이어 건너뛴 hidden(→final norm→lm_head)으로 뽑은 다음 토큰이
full 모델 argmax와 자주 일치하면, 그 truncated forward를 draft로 쓰고 full로 verify(rejection
sampling=출력동등) → decode 연산 절감. accept 천장 = argmax 일치율.
신규성: vLLM은 EAGLE/Medusa(별도 draft)는 있어도 self layer-skip draft 없음. GPU-direct.
측정: 8B(빠른 프록시)로 skip 깊이별 top1 일치율 + draft FLOP 절감률.
"""
import torch, os, time
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
m=model.model; nl=len(m.layers)
print(f"layers={nl}", flush=True)

texts=[
 "The capital of France is Paris, and the capital of Japan is Tokyo. The largest planet in the solar system is",
 "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n# test\nprint(fibonacci(10))\n",
 "In machine learning, gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by",
 "Once upon a time, in a small village nestled between two mountains, there lived a young blacksmith who dreamed of",
]
@torch.no_grad()
def hidden_states_all(ids):
    out=model(ids, output_hidden_states=True)
    return out.hidden_states, out.logits  # hidden_states: tuple len nl+1 (embed + each layer)
def logits_from_hidden(h):
    h=m.norm(h)
    return model.lm_head(h)

skip_list=[2,4,8,12,16]
print("skip(마지막N층 생략) | top1일치율 | draft연산비율(=(nl-N)/nl)", flush=True)
agg={}
for ids_text in texts:
    ids=tok(ids_text,return_tensors="pt").input_ids.to(dev)
    hs, full_logits=hidden_states_all(ids)
    full_arg=full_logits[0].argmax(-1)  # [seq]
    for N in skip_list:
        depth=nl-N  # hidden_states[depth] = depth개 레이어 통과 후 (index 0=embed)
        h=hs[depth][0]
        lg=logits_from_hidden(h)
        arg=lg.argmax(-1)
        # 다음토큰 예측 일치 = 각 위치 argmax 비교 (teacher-forcing, 모든 위치)
        match=(arg==full_arg).float().mean().item()
        agg.setdefault(N,[]).append(match)
print(flush=True)
for N in skip_list:
    mr=sum(agg[N])/len(agg[N]); comp=(nl-N)/nl
    print(f"  skip {N:2d} (depth {nl-N:2d}) | top1={mr:.3f} | draft연산={comp:.2f} | 예상 net = accept×절감 휴리스틱", flush=True)
print("\n판정: 마지막 N층 skip의 top1 일치율이 높을수록(>0.6) self-spec accept 천장↑.", flush=True)
print("높은 accept × 큰 N(연산절감) 조합이 있으면 GO(vLLM 부분forward proposer 구현). 낮으면 기각.", flush=True)
