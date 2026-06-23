"""iter2 신규방향 feasibility: 저정밀(FP8) all-reduce 정밀도 — R1-671B 통신-bound(81%) 직격.
8-rank all-reduce를 bf16(기준) vs FP8-RTN vs FP8-stochastic-rounding vs FP8+per-channel-scale로
시뮬해 합산 상대오차 측정. SR(unbiased)이 RTN 이기는지 = 신규 알고리즘 핵심 가설.
통신량: bf16 2B → FP8 1B = 2× 절감. 오차가 분포동등 게이트 수준이면 viable.
"""
import torch
torch.manual_seed(0); dev="cuda:0"
W=8  # ranks
H=7168  # R1 hidden
N=4096  # tokens
def to_e4m3(x, scale):
    q=(x/scale).clamp(-448,448).to(torch.float8_e4m3fn)
    return q.to(torch.float32)*scale
def to_e4m3_sr(x, scale):  # stochastic rounding
    y=x/scale
    lo=y.to(torch.float8_e4m3fn).to(torch.float32)
    # 다음 표현가능값 근사: e4m3 간격~상대적, SR을 단순화(확률적 반올림 근사)
    err=y-lo
    # bump: 50%*|err| 방향 확률반올림 (근사)
    bump=(torch.rand_like(y)<err.abs()).to(torch.float32)*torch.sign(err)
    step=(lo*0+1)*(scale*0+1)  # placeholder
    return (lo+bump*0)*scale + err*scale*0 + lo*0  # SR 정확 구현 어려움 → 아래 직접

def fp8_rtn(xs, per_channel):
    if per_channel:
        s=torch.stack([x.abs().amax(0,keepdim=True).clamp_min(1e-6)/448 for x in xs])
    else:
        s=torch.stack([x.abs().amax().clamp_min(1e-6)/448 for x in xs]).view(W,1,1)
    return sum(to_e4m3(xs[i], s[i]) for i in range(W))
def fp8_sr(xs, per_channel):
    # 정확한 SR: y=x/scale, p=frac to next; torch float8 직접 SR 미지원→근사: round + noise injection
    out=0
    for i in range(W):
        sc=(xs[i].abs().amax(0,keepdim=True) if per_channel else xs[i].abs().amax()).clamp_min(1e-6)/448
        y=xs[i]/sc
        rn=y.to(torch.float8_e4m3fn).to(torch.float32)
        # dithered: 양자화 전 작은 균등노이즈로 SR 근사(반올림 편향 제거)
        dith=(torch.rand_like(y)-0.5)*(y.abs().clamp_min(1e-6)*0.06)  # ~e4m3 상대간격
        rs=(y+dith).to(torch.float8_e4m3fn).to(torch.float32)
        out=out+rs*sc
    return out

print(f"8-rank all-reduce 정밀도 (H={H}, N={N}, R1 hidden)\n분포: Gaussian + 5% outlier(×10)")
for trial in range(3):
    xs=[]
    for _ in range(W):
        x=torch.randn(N,H,device=dev,dtype=torch.float32)
        mask=torch.rand(N,H,device=dev)<0.05; x=x+mask*torch.randn_like(x)*10  # outlier
        xs.append(x)
    ref=sum(xs)  # bf16급 기준(여기선 fp32 기준)
    refn=ref.norm()
    for name,fn,pc in [("FP8-RTN(per-tensor)",fp8_rtn,False),("FP8-RTN(per-chan)",fp8_rtn,True),
                       ("FP8-SR(per-tensor)",fp8_sr,False),("FP8-SR(per-chan)",fp8_sr,True)]:
        out=fn(xs,pc)
        rel=(out-ref).norm()/refn
        maxd=(out-ref).abs().max().item()
        print(f"  [{trial}] {name:22s} rel_err={rel.item():.4f}  max_abs={maxd:.3f}")
print("\n판정: rel_err 작고(<0.01?) SR이 RTN 이기면 → FP8 AR viable, 신규 오차보상 알고리즘 설계.")
print("통신 2× 절감(bf16→FP8). 게이트는 출력 logprob 영향으로 별도 검증 필요.")
