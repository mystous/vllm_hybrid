"""bump-mixed NVFP4 70B 체크포인트 생성 (llm-compressor oneshot).
probe70b.json의 민감 블록 top-N을 ignore(→bf16 고정밀 유지)하고 나머지는 NVFP4 W4A4.
usage: python make_mixed.py <n_bump_blocks> <out_dir>
"""
import sys, json, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from datasets import load_dataset

N_BUMP=int(sys.argv[1]) if len(sys.argv)>1 else 2
OUT=sys.argv[2] if len(sys.argv)>2 else "/raid/hf_cache/mixed_nvfp4_70b"
HF="meta-llama/Llama-3.1-70B-Instruct"

pj=json.load(open("runs/probe70b.json"))
order=pj["order"]            # 민감 순 (KL 큰 순) 블록 라벨 "L0-9" ...
bump_blocks=order[:N_BUMP]
ignore=["lm_head"]
for b in bump_blocks:
    lo,hi=b[1:].split("-");
    for li in range(int(lo),int(hi)+1):
        ignore.append(f"re:model\\.layers\\.{li}\\..*")
print(f"bump blocks(고정밀 유지)={bump_blocks} → ignore {len(ignore)-1} layer regex", flush=True)

tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype="auto",device_map="auto")
NS=256; ML=512
try:
    raw=load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft").shuffle(seed=42).select(range(NS))
    def pp(ex): return {"text":tok.apply_chat_template(ex["messages"],tokenize=False)}
    raw=raw.map(pp)
    def tk(ex): return tok(ex["text"],padding=False,truncation=True,max_length=ML,add_special_tokens=False)
    ds=raw.map(tk,remove_columns=raw.column_names)
    print(f"calibration = ultrachat_200k {NS} samples", flush=True)
except Exception as e:
    print(f"[WARN] ultrachat 로드 실패({type(e).__name__}: {str(e)[:80]}) → 로컬 합성 calibration 폴백", flush=True)
    from datasets import Dataset
    import itertools
    topics=["machine learning systems","quantum chemistry","economic policy","ancient roman history",
            "distributed databases","climate modeling","protein folding","compiler optimization",
            "music theory","ocean currents","graph algorithms","tax law","neural rendering","epidemiology"]
    seeds=[f"Explain in detail how {t} works, with concrete examples and step-by-step reasoning. "
           f"Then discuss common misconceptions about {t} and how practitioners address them in real systems. " for t in topics]
    texts=[ (seeds[i%len(seeds)]*6) for i in range(NS) ]
    def tk2(s): return tok(s,padding=False,truncation=True,max_length=ML,add_special_tokens=True)
    ds=Dataset.from_list([tk2(t) for t in texts])
    print(f"calibration = 로컬 합성 {NS} samples", flush=True)

recipe=QuantizationModifier(targets="Linear",scheme="NVFP4",ignore=ignore)
oneshot(model=model,dataset=ds,recipe=recipe,max_seq_length=ML,num_calibration_samples=NS,output_dir=OUT)
print(f"saved mixed checkpoint -> {OUT}", flush=True)
