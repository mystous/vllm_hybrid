"""Round3: AWQ + GPTQ 결합 NVFP4 W4A4 70B.
R1 GPTQ(ppl_rel↓/max_diff↑)·R2 AWQ(max_diff↓/ppl_rel↑)가 직교 → 결합으로 양축 동시 해결 시도.
AWQ 채널보호 스케일 → GPTQ 오차역전파, 둘 다 NVFP4 W4A4.
usage: python make_awqgptq.py <out_dir>
"""
import sys, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset, Dataset
OUT=sys.argv[1] if len(sys.argv)>1 else "/raid/hf_cache/awqgptq_nvfp4_70b"
HF="meta-llama/Llama-3.1-70B-Instruct"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype="auto",device_map="auto")
NS=512; ML=512
try:
    raw=load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft").shuffle(seed=42).select(range(NS))
    raw=raw.map(lambda ex:{"text":tok.apply_chat_template(ex["messages"],tokenize=False)})
    ds=raw.map(lambda ex:tok(ex["text"],padding=False,truncation=True,max_length=ML,add_special_tokens=False),
               remove_columns=raw.column_names)
    print(f"calibration = ultrachat_200k {NS}", flush=True)
except Exception as e:
    print(f"[WARN] ultrachat 실패({type(e).__name__}) → 합성 폴백", flush=True)
    topics=["machine learning systems","quantum chemistry","economic policy","ancient roman history",
            "distributed databases","climate modeling","protein folding","compiler optimization",
            "music theory","ocean currents","graph algorithms","tax law","neural rendering","epidemiology"]
    seeds=[f"Explain in detail how {t} works, with concrete examples and step-by-step reasoning. "
           f"Then discuss common misconceptions about {t} and how practitioners address them. " for t in topics]
    texts=[(seeds[i%len(seeds)]*6) for i in range(NS)]
    ds=Dataset.from_list([tok(t,padding=False,truncation=True,max_length=ML,add_special_tokens=True) for t in texts])
    print(f"calibration = 합성 {NS}", flush=True)
# AWQ(채널 saliency 스케일) → GPTQ(오차역전파), 둘 다 NVFP4. AWQ가 scheme 적용, GPTQ가 동일 scheme 오차보정.
recipe=[
    AWQModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"]),
    GPTQModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"]),
]
oneshot(model=model,dataset=ds,recipe=recipe,max_seq_length=ML,num_calibration_samples=NS,output_dir=OUT)
print(f"saved AWQ+GPTQ-NVFP4 checkpoint -> {OUT}", flush=True)
