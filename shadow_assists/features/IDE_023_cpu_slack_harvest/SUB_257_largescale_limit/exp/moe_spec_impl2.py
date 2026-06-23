"""MoE self-spec 정확 구현 v2 — 별도 KV 2개 + DynamicCache.crop. 출력동등 + wall-clock.
draft(top-1) KV ≠ target(top-2) KV. verify 후 둘 다 accepted 위치로 crop. 출력=top-2 greedy 동등.
"""
import torch, os, time
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
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

def crop(cache, L):
    if hasattr(cache,"crop"): cache.crop(L); return cache
    for i in range(len(cache.key_cache)):
        cache.key_cache[i]=cache.key_cache[i][:,:,:L,:]; cache.value_cache[i]=cache.value_cache[i][:,:,:L,:]
    return cache

@torch.no_grad()
def baseline(ids, G):
    DK[0]=2; c=DynamicCache()
    o=model(ids,past_key_values=c,use_cache=True); nx=o.logits[:,-1:].argmax(-1); out=[nx.item()]
    for _ in range(G-1):
        o=model(nx,past_key_values=o.past_key_values,use_cache=True); nx=o.logits[:,-1:].argmax(-1); out.append(nx.item())
    return out

@torch.no_grad()
def spec(ids, G, K=4):
    plen=ids.shape[1]
    # prefill 둘 다 (draft top-1 KV, target top-2 KV)
    DK[0]=2; tc=DynamicCache(); ot=model(ids,past_key_values=tc,use_cache=True); cur=ot.logits[:,-1:].argmax(-1)
    DK[0]=1; dc=DynamicCache(); model(ids,past_key_values=dc,use_cache=True)
    out=[cur.item()]; pos=plen  # target KV가 커버하는 토큰 수(prompt). cur는 pos번째(미커밋)
    nacc=0; ncy=0
    while len(out)<G:
        ncy+=1
        # draft: cur부터 top-1로 K토큰, draft KV(dc) 진행. dc는 prompt까지 → cur 먼저 넣어야.
        DK[0]=1; d=[]; dn=cur
        for _ in range(K):
            od=model(dn,past_key_values=dc,use_cache=True); dn=od.logits[:,-1:].argmax(-1); d.append(dn.item())
        # 이제 dc는 pos + 1(cur) + K... 실제로 cur 포함 K+1 진행됨? cur 1 + K-1 내부? 위 루프 K번 → cur,d0..d_{K-2} 입력, d0..d_{K-1} 출력. dc 길이 = pos+K.
        # verify: target에 [cur, d0..d_{K-2}] (K개) 입력 → 각 위치 argmax
        DK[0]=2; vin=torch.tensor([[cur.item()]+d[:-1]],device=dev)
        ov=model(vin,past_key_values=tc,use_cache=True); varg=ov.logits[0].argmax(-1).tolist()  # K개
        # tc는 이제 pos+K 커버. accept: d[i]==varg[i]
        a=0
        for i in range(K):
            if d[i]==varg[i]: a+=1
            else: break
        committed=d[:a]+[varg[a] if a<K else varg[K-1]]  # a matched + bonus(varg[a])
        for t in committed:
            if len(out)<G: out.append(t)
        nacc+=a
        # KV crop (off-by-one 수정): verify가 cur의 KV를 pos에 추가 → cur(pos)+d0..d_{a-1}(pos+1..pos+a)
        # = pos+a+1 유지. bonus(varg[a])는 KV 미생성→다음 cur로 재처리.
        newpos=pos+a+1
        crop(tc, newpos); crop(dc, newpos)
        cur=torch.tensor([[committed[-1]]],device=dev); pos=newpos
    return out[:G], nacc/max(ncy,1)

prompts=["The history of artificial intelligence began","def merge_sort(arr):","The economic implications of climate change include"]
G=64
print("MoE self-spec v2 실측 (별도 KV + crop)")
ok=True
for pr in prompts:
    ids=tok(pr,return_tensors="pt").input_ids.to(dev); baseline(ids,4)
    torch.cuda.synchronize(); t0=time.perf_counter(); b=baseline(ids,G); tb=time.perf_counter()-t0
    torch.cuda.synchronize(); t0=time.perf_counter(); s,acc=spec(ids,G,4); ts=time.perf_counter()-t0
    match=sum(1 for x,y in zip(b,s) if x==y)/G
    if match<0.99: ok=False
    print(f"  base {G/tb:.1f}tok/s | spec {G/ts:.1f}tok/s → {tb/ts:.2f}× | accept/cycle={acc:.2f} | 출력일치={match:.3f}")
print(f"\n출력동등={'OK' if ok else 'FAIL(버그)'}. 판정: 동등 AND speedup>1.1 = win.")
