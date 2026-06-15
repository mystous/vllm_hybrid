#!/usr/bin/env python3
"""KTAIL 오프라인 검증 — 동적 throughput 컨트롤러를 기존 252셀 측정 로그로 평가.
GPU 불요. 데이터: runs_multimodel/summ_<model>_k<K>pad_<corpus>.json
각 파일 = (model, K, corpus) 셀: output_tps, accept_rate(α), accept_tokens, draft_tokens.
"""
import glob, json, os, re, math
from collections import defaultdict

D = os.path.join(os.path.dirname(__file__), "runs_multimodel")
KSET = [4, 6, 8, 12]

# cell[(model,corpus)][K] = dict(tps, alpha, acc, draft)
cell = defaultdict(dict)
for f in glob.glob(os.path.join(D, "summ_*.json")):
    m = re.match(r"summ_(.+)_k(\d+)pad_(.+)\.json$", os.path.basename(f))
    if not m:
        continue
    model, K, corpus = m.group(1), int(m.group(2)), m.group(3)
    d = json.load(open(f))
    cell[(model, corpus)][K] = dict(
        tps=d.get("output_tps") or 0.0,
        alpha=d.get("accept_rate") or 0.0,
        acc=d.get("accept_tokens") or 0.0,
        draft=d.get("draft_tokens") or 0.0,
        toks=d.get("total_completion_tokens") or 0.0,
    )

cells = {k: v for k, v in cell.items() if all(K in v for K in KSET)}
print(f"전체 (model,corpus) 셀: {len(cell)}  /  4-K 완비 셀: {len(cells)}")

# 셀별: oracle-K, 고정 K6, alpha(중앙), coverage s 추정
def coverage(c, K):
    # draft_tokens = K * num_draft_steps  → num_draft_steps = draft/K
    # total_steps ≈ total_tokens / mean_accept_len ; mean_accept_len = 1 + accept/draft_steps... 근사
    draft_steps = c["draft"] / K if K else 0
    mal = 1 + (c["acc"] / draft_steps) if draft_steps else 1
    total_steps = c["toks"] / mal if mal else 0
    return (draft_steps / total_steps) if total_steps else 0.0, mal

rows = []
for (model, corpus), v in sorted(cells.items()):
    tps = {K: v[K]["tps"] for K in KSET}
    oracle_K = max(KSET, key=lambda K: tps[K])
    alpha = v[8]["alpha"]  # α는 K에 거의 불변 — K8 기준
    s8, mal8 = coverage(v[8], 8)
    rows.append(dict(model=model, corpus=corpus, tps=tps, oracleK=oracle_K,
                     alpha=alpha, s=s8, mal=mal8,
                     tps_oracle=tps[oracle_K], tps_k6=tps[6]))

# 요약: oracle vs 고정K6 (기존 +38% vs +49% 재현 확인)
import statistics as st
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
print("\n=== oracle-K 분포 ===")
from collections import Counter
print(Counter(r["oracleK"] for r in rows))
print("\n=== α vs oracle-K (정렬) ===")
for r in sorted(rows, key=lambda r: r["alpha"]):
    print(f"  {r['model'][:28]:28s} {r['corpus']:10s} α={r['alpha']:.3f} s={r['s']:.3f} mal={r['mal']:.2f} → oracleK={r['oracleK']:2d}  (tps K4/6/8/12={r['tps'][4]:.0f}/{r['tps'][6]:.0f}/{r['tps'][8]:.0f}/{r['tps'][12]:.0f})")

# 각 셀에서 K6 대비 oracle 의 상방
gains = [r["tps_oracle"] / r["tps_k6"] for r in rows if r["tps_k6"] > 0]
print(f"\noracle/K6 기하평균 = {geo(gains):.4f}  (셀별 oracle 선택의 K6 대비 상방)")

# ============ 핵심 수치: oracle vs 고정K ============
print("\n" + "="*60)
KSETv = KSET
# 고정 글로벌 best-K (모든 셀 평균 tps 최대인 단일 K)
import statistics as _st
fixed_tps = {K: geo([r["tps"][K] for r in rows if r["tps"][K]>0]) for K in KSETv}
best_global_K = max(KSETv, key=lambda K: fixed_tps[K])
print(f"고정 단일 K 기하평균 tps: " + " ".join(f"K{K}={fixed_tps[K]:.0f}" for K in KSETv))
print(f"→ 고정 best-global K = K{best_global_K}")
oracle_over_k6   = geo([r["tps_oracle"]/r["tps"][6]  for r in rows if r["tps"][6]>0])
oracle_over_best = geo([r["tps_oracle"]/r["tps"][best_global_K] for r in rows if r["tps"][best_global_K]>0])
print(f"oracle/K6        기하평균 = {oracle_over_k6:.4f}")
print(f"oracle/best-global(K{best_global_K}) 기하평균 = {oracle_over_best:.4f}  ← 동적이 노릴 상방")

# ============ α 비예측성 정량: 최적 단조 α→K 규칙의 regret ============
print("\n=== α-버킷의 한계 (최적 단조 threshold 규칙조차) ===")
# α 정렬 후 3개 cutpoint로 {4,6,8,12} 할당하는 최적 규칙 brute-force → 그 regret
import itertools
rs = sorted(rows, key=lambda r:r["alpha"])
alphas=[r["alpha"] for r in rs]
def rule_tps(cuts):
    # cuts=(c1,c2,c3) → α<c1:K4, <c2:K6, <c3:K8, else K12
    tot=[]
    for r in rs:
        a=r["alpha"]
        K = 4 if a<cuts[0] else 6 if a<cuts[1] else 8 if a<cuts[2] else 12
        tot.append(r["tps"][K]/r["tps_oracle"])
    return geo(tot)
best=0;bc=None
grid=[alphas[i] for i in range(0,len(alphas),2)]
for c in itertools.combinations(grid,3):
    g=rule_tps(c)
    if g>best: best,bc=g,c
print(f"최적 단조 α-threshold 규칙의 oracle 회수율 = {best:.4f}  (1.0=oracle)")
print(f"  → α만으로는 oracle의 {best*100:.1f}% 만 — 나머지는 α-무관(모델/corpus 구조).")

# ============ 동적 throughput-feedback 컨트롤러 시뮬 (워크로드 시프트) ============
print("\n=== 동적 컨트롤러 시뮬 (단일 모델, corpus 시프트 스트림) ===")
def simulate(model, seq, eps=0.15, win=1):
    """UCB-식: 각 corpus 구간에서 측정 tps로 K 선택. 시프트 감지 시 재탐색.
       reward = 해당 (model,corpus,K) 의 실측 tps. policy 별 누적 tps 평균 반환."""
    corp_tps = {}  # (corpus,K)->tps
    for (m,c),v in cells.items():
        if m==model:
            for K in KSET: corp_tps[(c,K)]=v[K]["tps"]
    import random; random.seed(0)
    # 컨트롤러: per-K running mean + count, 시프트 시 reset
    res={}
    for pol in ["dynamic","fixedK6","fixed_best","oracle"]:
        mean={K:0.0 for K in KSET}; cnt={K:0 for K in KSET}
        total=0.0; n=0; prev=None; curK=KSET[0]
        for c in seq:
            if c not in [x[0] for x in corp_tps]: continue
            if pol=="oracle":
                K=max(KSET,key=lambda K:corp_tps[(c,K)])
            elif pol=="fixedK6": K=6
            elif pol=="fixed_best": K=best_global_K
            else: # dynamic
                if c!=prev:  # 시프트 → 탐색 리셋
                    mean={K:0.0 for K in KSET}; cnt={K:0 for K in KSET}
                import math as _m
                # UCB1
                t=sum(cnt.values())+1
                def ucb(K): return (mean[K] if cnt[K] else 1e9) + (_m.sqrt(2*_m.log(t)/cnt[K]) * (max(corp_tps[(c,kk)] for kk in KSET)) if cnt[K] else 0)
                K=max(KSET,key=ucb)
                r=corp_tps[(c,K)]; cnt[K]+=1; mean[K]+=(r-mean[K])/cnt[K]
            total+=corp_tps[(c,K)]; n+=1; prev=c
        res[pol]=total/n if n else 0
    return res

# 시프트 스트림: corpus 를 구간별로 (각 corpus 10회 머무르고 전환), 두 모델 예시
import random
corpora=["mbpp","sharegpt","mix","lmsys","mix","humaneval","swebench","mix"]
seq=[]
for c in corpora: seq+=[c]*10
for model in ["DeepSeek-R1-Distill-Qwen-32B","Qwen2.5-7B-Instruct","DeepSeek-R1-Distill-Qwen-7B"]:
    r=simulate(model,seq)
    print(f"  {model[:30]:30s} dyn={r['dynamic']:.0f} fixedK6={r['fixedK6']:.0f} fixed_best(K{best_global_K})={r['fixed_best']:.0f} oracle={r['oracle']:.0f}"
          f"  | dyn/oracle={r['dynamic']/r['oracle']*100:.1f}% dyn/K6={r['dynamic']/r['fixedK6']*100:.1f}%")

print("\n=== 레짐 길이별 dynamic/oracle (탐색 상각) ===")
for reglen in [5,20,100,500]:
    seq2=[]
    for c in corpora: seq2+=[c]*reglen
    accs=[]
    for model in ["DeepSeek-R1-Distill-Qwen-32B","Qwen2.5-7B-Instruct","DeepSeek-R1-Distill-Qwen-7B","Qwen2.5-32B-Instruct"]:
        r=simulate(model,seq2)
        accs.append((r['dynamic']/r['oracle'], r['dynamic']/r['fixedK6']))
    do=geo([a[0] for a in accs]); dk=geo([a[1] for a in accs])
    print(f"  레짐길이 {reglen:4d}: dyn/oracle={do*100:.1f}%  dyn/K6={dk*100:.1f}%")
