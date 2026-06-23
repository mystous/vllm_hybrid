"""MoE self-speculative decoding 실제 구현 + wall-clock 속도 측정 (probe 아님).
draft=top-1(gate 절단)로 K토큰 autoregressive 생성 → verify=top-2 단일 forward로 검증
→ greedy rejection(top-2 argmax와 일치까지 accept) = 출력동등(top-2 greedy와 동일 시퀀스).
측정: spec tok/s vs baseline(top-2 autoregressive) tok/s. 실제 speedup.
"""
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

@torch.no_grad()
def baseline(ids, G):
    DK[0]=2
    o=model(ids,use_cache=True); past=o.past_key_values; nx=o.logits[:,-1:].argmax(-1)
    out=[nx.item()]
    for _ in range(G-1):
        o=model(nx,past_key_values=past,use_cache=True); past=o.past_key_values; nx=o.logits[:,-1:].argmax(-1); out.append(nx.item())
    return out

@torch.no_grad()
def spec_decode(ids, G, K=4):
    # target KV로 진행. 매 사이클: draft가 K토큰 생성(top-1, 자체 KV) → target이 [last]+draft K개를
    # 한 forward로 검증 → greedy 일치까지 accept → target KV를 accepted까지 진행.
    DK[0]=2
    o=model(ids,use_cache=True); tgt_past=o.past_key_values; cur=o.logits[:,-1:].argmax(-1)  # 첫 토큰
    out=[cur.item()]; ncycle=0; nacc=0
    while len(out)<G:
        ncycle+=1
        # draft: top-1로 cur에서 K토큰 (target KV 복제 대신, draft는 가벼운 별도 진행)
        DK[0]=1
        # draft는 target과 동일 KV 위치에서 시작해야 → draft용 forward는 cur 1토큰 + 직전 context KV 필요.
        # 단순·정확 구현: draft도 target KV를 공유(같은 위치). draft가 cur로 forward(top-1) K번.
        d_ids=[]; dpast=tgt_past; dn=cur
        dlogits=[]
        for _ in range(K):
            o=model(dn,past_key_values=dpast,use_cache=True); dpast=o.past_key_values
            dn=o.logits[:,-1:].argmax(-1); d_ids.append(dn.item())
        # verify: target(top-2)이 [cur, d0..d_{K-2}] 입력으로 K개 위치 검증 (각 위치의 다음 argmax)
        DK[0]=2
        vin=torch.tensor([[cur.item()]+d_ids[:-1]],device=dev)
        ov=model(vin,past_key_values=tgt_past,use_cache=True)
        varg=ov.logits[0].argmax(-1).tolist()  # K개: 각 위치 target 예측
        # accept: draft d_i가 target 예측 varg[i]와 일치하는 동안
        acc=0
        for i in range(K):
            if d_ids[i]==varg[i]: acc+=1
            else: break
        # accepted 토큰 + 1 bonus(target의 다음) = varg[:acc]는 d_ids[:acc]와 동일, 그 다음 토큰=varg[acc]
        accepted=varg[:acc]+[varg[acc] if acc<K else varg[K-1]]  # bonus
        for t in accepted:
            if len(out)<G: out.append(t)
        nacc+=acc
        # target KV를 accepted 길이만큼 진행: vin의 앞 (acc+1)개를 target KV에 커밋해야.
        # 정확·단순: target KV를 재계산(accepted 반영). 비용 절감 위해 ov의 past를 acc+1까지 잘라씀.
        commit=torch.tensor([[cur.item()]+accepted[:-1]][0][:acc+1],device=dev).unsqueeze(0) if acc>=0 else None
        oc=model(commit,past_key_values=tgt_past,use_cache=True)
        tgt_past=oc.past_key_values; cur=torch.tensor([[accepted[-1]]],device=dev)
        out_last=accepted[-1]
    return out[:G], nacc/max(ncycle,1)

prompts=["The history of artificial intelligence began","def merge_sort(arr):","The economic implications of climate change include"]
G=64
print("MoE self-spec 실측 (Mixtral, top-1 draft / top-2 verify)")
for pr in prompts:
    ids=tok(pr,return_tensors="pt").input_ids.to(dev)
    # warm
    baseline(ids,4)
    torch.cuda.synchronize(); t0=time.perf_counter(); b=baseline(ids,G); tb=time.perf_counter()-t0
    torch.cuda.synchronize(); t0=time.perf_counter(); s,acc=spec_decode(ids,G,K=4); ts=time.perf_counter()-t0
    match=sum(1 for x,y in zip(b,s) if x==y)/G
    print(f"  base {tb*1000:.0f}ms({G/tb:.1f}tok/s) | spec {ts*1000:.0f}ms({G/ts:.1f}tok/s) → {tb/ts:.2f}× | accept/cycle={acc:.2f} | 출력일치={match:.3f}")
print("\n판정: spec speedup>1.1 AND 출력일치≈1.0(출력동등)이면 MoE-spec 실구현 win.")
