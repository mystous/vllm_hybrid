"""R4 go/no-go CPU 마이크로벤치: sketch/hash 기반 ngram draft vs vLLM 현행 KMP 정확매칭.
신규성 = vLLM에 sketch-draft 미존재(현행은 ngram_proposer.py의 KMP O(N) 재스캔 + suffix arctic tree).
측정: (a) draft 생성 지연(per call), (b) accept 프록시(제안 토큰이 실제 다음 토큰과 일치하는 비율).
데이터: 반복성 있는 실 토큰열(허깅페이스 토크나이저로 코드/대화 코퍼스 토큰화, 없으면 합성 반복열).
"""
import time, numpy as np, os
from collections import defaultdict, deque

# ---------- 현행 vLLM ngram: KMP 최장-suffix 매칭 (ngram_proposer.py:626-713 발췌·동등) ----------
def kmp_longest_match_propose(origin_tokens, min_ngram, max_ngram, k):
    total = origin_tokens.shape[0]
    if total < min_ngram: return np.empty(0, dtype=np.int64)
    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest=0; position=0; prev_lps=0; i=1
    while i < total:
        if tokens[prev_lps]==tokens[i]:
            prev_lps+=1
            if prev_lps>=longest: longest=prev_lps; position=i
            if i<max_ngram: lps[i]=prev_lps
            if prev_lps==max_ngram: prev_lps=lps[max_ngram-1]
            i+=1
        elif prev_lps!=0: prev_lps=lps[prev_lps-1]
        else: i+=1
    if longest<min_ngram: return np.empty(0,dtype=np.int64)
    start=total-1-position+longest
    kk=min(k,total-start)
    return origin_tokens[start:start+kk]

# ---------- R4 후보: 증분 hash-table ngram LM draft (O(1) lookup/update) ----------
class HashNgramDrafter:
    """마지막 n토큰 → 다음토큰 빈도 카운터. 토큰 생성 시 O(1) 갱신, draft 시 O(1) 조회.
    KMP의 매 호출 O(N) 재스캔을 제거. (CMS는 메모리 상한판; 여기선 정확 dict로 상한 비교)"""
    def __init__(self, n, k):
        self.n=n; self.k=k
        self.table=defaultdict(lambda: defaultdict(int))  # key(tuple n토큰)->{next:count}
    def update(self, ctx, nxt):
        if len(ctx)>=self.n:
            key=tuple(ctx[-self.n:]); self.table[key][nxt]+=1
    def propose(self, ctx):
        if len(ctx)<self.n: return []
        key=tuple(ctx[-self.n:])
        out=[]; cur=deque(key, maxlen=self.n)
        for _ in range(self.k):
            d=self.table.get(tuple(cur))
            if not d: break
            nx=max(d, key=d.get)          # 최빈 다음토큰 (greedy chain)
            out.append(nx); cur.append(nx)
        return out

class MultiOrderHashDrafter:
    """여러 n(min..max)의 hash 테이블 → draft 시 최장 n부터 조회(KMP 최장매칭 정책 O(1) 재현).
    accept 품질=KMP 동등 목표, 지연=O(max_n) 해시조회(재스캔 O(N) 제거)."""
    def __init__(self, n_lo, n_hi, k):
        self.n_lo=n_lo; self.n_hi=n_hi; self.k=k
        self.tables={n: defaultdict(lambda: defaultdict(int)) for n in range(n_lo,n_hi+1)}
    def update(self, ctx, nxt):
        L=len(ctx)
        for n in range(self.n_lo,self.n_hi+1):
            if L>=n: self.tables[n][tuple(ctx[-n:])][nxt]+=1
    def propose(self, ctx):
        out=[]; cur=list(ctx)
        for _ in range(self.k):
            nx=None
            for n in range(self.n_hi,self.n_lo-1,-1):   # 최장 n 우선
                if len(cur)<n: continue
                d=self.tables[n].get(tuple(cur[-n:]))
                if d: nx=max(d,key=d.get); break
            if nx is None: break
            out.append(nx); cur.append(nx)
        return out

def load_token_streams():
    os.environ["HF_HOME"]="/raid/hf_cache"
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        # 반복성 큰 실텍스트(코드/구조 반복) — draft가 의미있는 regime
        texts=[
            ("def process(items):\n    for item in items:\n        if item.valid:\n            result.append(item.value)\n    return result\n")*8,
            ("The quarterly report shows revenue growth. The quarterly report shows margin growth. The quarterly report shows user growth. ")*8,
            ("import numpy as np\narr = np.zeros((10,10))\nfor i in range(10):\n    for j in range(10):\n        arr[i][j] = i*j\n")*8,
        ]
        return [np.array(tok(t).input_ids,dtype=np.int64) for t in texts]
    except Exception as e:
        print(f"[WARN] tokenizer 실패({type(e).__name__}) → 합성 반복열");
        rng=np.random.RandomState(0)
        base=rng.randint(0,5000,size=60)
        return [np.tile(base, 10)+rng.randint(0,2,size=600) for _ in range(3)]

def evaluate(stream, n_lo=3, n_hi=8, k=6, warmup=64):
    """스트림을 토큰단위로 진행하며 각 위치에서 draft 생성·accept 측정."""
    res={}
    # --- KMP (현행) ---
    t0=time.perf_counter(); acc_kmp=0; tot_kmp=0; calls=0
    for pos in range(warmup, len(stream)-1):
        ctx=stream[:pos]
        draft=kmp_longest_match_propose(ctx, n_lo, n_hi, k); calls+=1
        # accept 프록시: 제안 토큰들이 실제 후속과 prefix 일치하는 길이
        actual=stream[pos:pos+len(draft)]
        m=0
        for a,b in zip(draft,actual):
            if a==b: m+=1
            else: break
        acc_kmp+=m; tot_kmp+=len(draft) if len(draft)>0 else 0
    t_kmp=time.perf_counter()-t0
    # --- Hash ngram (후보) ---
    dr=HashNgramDrafter(n_lo, k)  # n=n_lo 사용(최단 신뢰 ngram)
    for i in range(min(warmup,len(stream)-1)): dr.update(stream[:i+1], stream[i+1])
    t0=time.perf_counter(); acc_h=0; tot_h=0
    for pos in range(warmup, len(stream)-1):
        ctx=stream[:pos]
        draft=dr.propose(ctx)
        actual=stream[pos:pos+len(draft)]
        m=0
        for a,b in zip(draft,actual):
            if a==b: m+=1
            else: break
        acc_h+=m; tot_h+=len(draft) if len(draft)>0 else 0
        dr.update(stream[:pos+1], stream[pos+1])  # 온라인 갱신
    t_h=time.perf_counter()-t0
    # --- Multi-order hash (accept 회복 변종) ---
    dr2=MultiOrderHashDrafter(n_lo, n_hi, k)
    for i in range(min(warmup,len(stream)-1)): dr2.update(stream[:i+1], stream[i+1])
    t0=time.perf_counter(); acc_m=0; tot_m=0
    for pos in range(warmup, len(stream)-1):
        ctx=stream[:pos]; draft=dr2.propose(ctx)
        actual=stream[pos:pos+len(draft)]; m=0
        for a,b in zip(draft,actual):
            if a==b: m+=1
            else: break
        acc_m+=m; tot_m+=len(draft) if len(draft)>0 else 0
        dr2.update(stream[:pos+1], stream[pos+1])
    t_m=time.perf_counter()-t0
    return {
        "calls":calls,
        "kmp_ms_per_call": t_kmp/calls*1000,
        "hash_ms_per_call": t_h/calls*1000,
        "multi_ms_per_call": t_m/calls*1000,
        "kmp_accept": acc_kmp/max(tot_kmp,1),
        "hash_accept": acc_h/max(tot_h,1),
        "multi_accept": acc_m/max(tot_m,1),
        "kmp_acc_tok": acc_kmp, "hash_acc_tok": acc_h, "multi_acc_tok": acc_m,
    }

if __name__=="__main__":
    streams=load_token_streams()
    print(f"streams={len(streams)}, lens={[len(s) for s in streams]}")
    agg=defaultdict(float)
    for si,s in enumerate(streams):
        r=evaluate(s)
        print(f"\n[stream {si}] calls={r['calls']}")
        print(f"  KMP(현행)  : {r['kmp_ms_per_call']:.4f} ms/call, accept={r['kmp_accept']:.3f} ({r['kmp_acc_tok']} tok)")
        print(f"  Hash-fix   : {r['hash_ms_per_call']:.4f} ms/call, accept={r['hash_accept']:.3f} ({r['hash_acc_tok']} tok)")
        print(f"  MultiOrder : {r['multi_ms_per_call']:.4f} ms/call, accept={r['multi_accept']:.3f} ({r['multi_acc_tok']} tok)")
        sp=r['kmp_ms_per_call']/max(r['multi_ms_per_call'],1e-9)
        print(f"  → MultiOrder speedup={sp:.1f}× , accept Δ={r['multi_accept']-r['kmp_accept']:+.3f}")
        agg["kmp_ms"]+=r['kmp_ms_per_call']; agg["multi_ms"]+=r['multi_ms_per_call']
        agg["kmp_acc"]+=r['kmp_accept']; agg["multi_acc"]+=r['multi_accept']
    n=len(streams)
    print(f"\n=== 평균 ===")
    print(f"  KMP        : {agg['kmp_ms']/n:.4f} ms/call, accept={agg['kmp_acc']/n:.3f}")
    print(f"  MultiOrder : {agg['multi_ms']/n:.4f} ms/call, accept={agg['multi_acc']/n:.3f}")
    print(f"  speedup={agg['kmp_ms']/max(agg['multi_ms'],1e-9):.1f}× , accept Δ={(agg['multi_acc']-agg['kmp_acc'])/n:+.3f}")
    print("\n판정 기준: speedup>2× AND accept Δ≥-0.05 → vLLM 통합·70B 측정 GO. 아니면 R4 기각.")
