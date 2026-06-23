import torch, os, inspect
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="mistralai/Mixtral-8x7B-Instruct-v0.1"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
blocks=[m for m in model.modules() if m.__class__.__name__=="MixtralSparseMoeBlock"]
b=blocks[0]
print("attrs:", [a for a in ["top_k","num_experts_per_tok","hidden_dim"] if hasattr(b,a)])
print("forward src 일부:")
print("\n".join(inspect.getsource(type(b).forward).splitlines()[:18]))
ids=tok("The capital of France is Paris and the largest planet is Jupiter",return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    for bb in blocks: bb.top_k=2
    l2=model(ids).logits[0].float()
    for bb in blocks: bb.top_k=1
    l1=model(ids).logits[0].float()
print(f"\n영구 top_k=1 vs 2: logit max_diff={(l1-l2).abs().max().item():.4f}  argmax일치={(l1.argmax(-1)==l2.argmax(-1)).float().mean().item():.3f}")
