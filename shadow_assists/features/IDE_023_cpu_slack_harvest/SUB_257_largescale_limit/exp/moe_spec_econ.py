"""top-1 vs top-2 forward 지연 직접측정 = draft 연산비 c. MoE-spec 경제성 결정타.
decode-like(batch1, KV) 단일토큰 forward 반복 측정."""
import torch, os, time
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="mistralai/Mixtral-8x7B-Instruct-v0.1"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
blocks=[m for m in model.modules() if m.__class__.__name__=="MixtralSparseMoeBlock"]
DK=[2]
for b in blocks:
    g=b.gate; of=g.forward
    def mk(of):
        def f(h):
            out=of(h)
            if DK[0]>=2: return out
            rl,w,idx=out; w1=w[...,:1]; w1=w1/w1.sum(-1,keepdim=True); return rl,w1,idx[...,:1]
        return f
    g.forward=mk(of)
ids=tok("The capital of France is Paris and the largest planet is Jupiter and",return_tensors="pt").input_ids.to(dev)
@torch.no_grad()
def decode_lat(dk,N=40):
    DK[0]=dk
    out=model(ids,use_cache=True); past=out.past_key_values; nxt=out.logits[:,-1:].argmax(-1)
    for _ in range(3): o=model(nxt,past_key_values=past,use_cache=True); past=o.past_key_values; nxt=o.logits[:,-1:].argmax(-1)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N):
        o=model(nxt,past_key_values=past,use_cache=True); past=o.past_key_values; nxt=o.logits[:,-1:].argmax(-1)
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/N*1000
l2=decode_lat(2); l1=decode_lat(1)
c=l1/l2; a=0.819
print(f"decode forward 지연: top-2(full)={l2:.2f}ms  top-1(draft)={l1:.2f}ms")
print(f"draft 연산비 c=l1/l2={c:.3f}, accept a={a}")
print(f"→ MoE-spec 속도배율 1/(c+(1-a)) = {1/(c+(1-a)):.2f}")
print("판정: >1.1 = MoE-spec 生(novel 구현가치) / <1 = 死")
