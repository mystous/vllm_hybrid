"""반복1: CPU-AMX draft 모델 decode 지연 측정 → GPU step(~0.7ms, 70B FP4 TP8) 오버랩 가능성.
spec draft가 CPU에서 GPU verify에 숨으려면 draft latency < GPU step. 못 따라가면 draft가
critical-path가 되어 net-negative (layer-skip R5와 같은 1/(c+(1-a)) 논리). AMX(amx_bf16) 활용.
"""
import torch, os, time
os.environ["HF_HOME"]="/raid/hf_cache"
torch.set_num_threads(int(os.environ.get("NTH","64")))
from transformers import AutoModelForCausalLM, AutoTokenizer
for HF in ["meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct"]:
    print(f"\n=== {HF} on CPU (bf16, AMX, threads={torch.get_num_threads()}) ===", flush=True)
    tok=AutoTokenizer.from_pretrained(HF)
    model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map="cpu").eval()
    ids=tok("The capital of France is Paris and the largest planet is", return_tensors="pt").input_ids
    with torch.no_grad():
        out=model(ids, use_cache=True); past=out.past_key_values
        nxt=out.logits[:,-1:].argmax(-1)
        # warmup decode
        for _ in range(3):
            o=model(nxt, past_key_values=past, use_cache=True); past=o.past_key_values; nxt=o.logits[:,-1:].argmax(-1)
        # 측정: 20 step 단일토큰 decode
        N=20; t0=time.perf_counter()
        for _ in range(N):
            o=model(nxt, past_key_values=past, use_cache=True); past=o.past_key_values; nxt=o.logits[:,-1:].argmax(-1)
        dt=(time.perf_counter()-t0)/N*1000
    print(f"  CPU decode 지연 = {dt:.2f} ms/token", flush=True)
    print(f"  GPU step(70B FP4 TP8) ≈ 0.7 ms → draft가 {dt/0.7:.1f}× 느림 → {'오버랩 가능' if dt<0.7 else 'critical-path (오버랩 불가)'}", flush=True)
    del model
print("\n판정: CPU draft 지연 < GPU step(0.7ms) 여야 spec 오버랩 win. 크게 느리면 CPU-모델-draft 死.", flush=True)
