import torch, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="mistralai/Mixtral-8x7B-Instruct-v0.1"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
TOPK=[2]
blocks=[m for m in model.modules() if m.__class__.__name__=="MixtralSparseMoeBlock"]
print("blocks",len(blocks),"orig top_k",blocks[0].top_k)
def patch(b):
    of=b.forward
    def f(h):
        b.top_k=TOPK[0]; return of(h)
    return f
for b in blocks: b.forward=patch(b)
ids=tok("The capital of France is Paris and the largest planet is Jupiter which",return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    TOPK[0]=2; l2=model(ids).logits[0].float()
    TOPK[0]=1; l1=model(ids).logits[0].float()
print("top_k after run:", blocks[0].top_k)
print(f"logit diff (top1 vs top2): max={ (l1-l2).abs().max().item():.4f}  mean={(l1-l2).abs().mean().item():.5f}  norm_rel={((l1-l2).norm()/l2.norm()).item():.5f}")
print(f"argmax 일치율={(l1.argmax(-1)==l2.argmax(-1)).float().mean().item():.3f}")
# 패치 확실히 먹나: top_k=2 두번 vs top_k 1
