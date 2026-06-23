"""특성화 논문 그림 6종 — 모두 본 세션 실측값(분칠 금지). no-GPU."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
D="runs/figs"; os.makedirs(D, exist_ok=True)
plt.rcParams.update({"figure.dpi":130,"font.size":10})

# (a) flamegraph vs nsys — all_reduce 비중 측정 방법론
fig,ax=plt.subplots(figsize=(5,3.2))
labels=["flamegraph\n(py-spy CPU)","nsys GPU\n(cudagraph)","nsys GPU\n(eager)"]
vals=[6,31,73]; bars=ax.bar(labels,vals,color=["#bbb","#4c78a8","#e45756"])
ax.set_ylabel("all-reduce 비중 (%)"); ax.set_title("(a) 통신 비중은 측정 방법이 가른다\nflamegraph가 GPU 병목을 12× 과소평가")
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+1,f"{v}%",ha="center")
fig.tight_layout(); fig.savefig(f"{D}/fig_a_flamegraph_vs_nsys.png"); plt.close()

# (b) top-8 vs top-2 EP 커널 — comm(allgather) 불변 vs compute permute 감소
fig,ax=plt.subplots(figsize=(6,3.4))
ker=["ncclAllGather\n(inter-GPU 통신)","ep_gather\n(local permute)","ep_scatter2\n(local permute)"]
t8=[345,647,692]; t2=[327,384,339]
x=np.arange(len(ker)); w=0.36
ax.bar(x-w/2,t8,w,label="top-8",color="#4c78a8")
ax.bar(x+w/2,t2,w,label="top-2",color="#f58518")
for i,(a,b) in enumerate(zip(t8,t2)):
    ax.text(i-w/2,a+8,f"{a}",ha="center",fontsize=8); ax.text(i+w/2,b+8,f"{b}",ha="center",fontsize=8)
    ax.text(i,max(a,b)+45,f"{b/a:.2f}×",ha="center",fontweight="bold",color="#d62728" if i==0 else "#2ca02c")
ax.set_xticks(x); ax.set_xticklabels(ker,fontsize=8); ax.set_ylabel("GPU 커널 시간 (M ns)")
ax.set_title("(b) 기본 allgather는 routing-invariant\n통신 0.95×(불변), 줄어든 건 local compute뿐"); ax.legend()
fig.tight_layout(); fig.savefig(f"{D}/fig_b_routing_invariant.png"); plt.close()

# (c) SPD layer{60} 게이트 위반
fig,axs=plt.subplots(1,2,figsize=(6,3.2))
for ax,metric,val,gate,nm in [(axs[0],"max_logprob_diff",1.288,0.5,"logprob"),(axs[1],"ppl_rel",0.199,0.1,"PPL")]:
    ax.bar([nm],[val],color="#e45756",width=0.5)
    ax.axhline(gate,color="k",ls="--",lw=1.5); ax.text(0,gate+0.02*max(val,gate),f"게이트 {gate}",ha="center",fontsize=8)
    ax.text(0,val+0.02*val,f"{val}",ha="center",fontweight="bold")
    ax.set_title(f"{metric}\n{val/gate:.1f}× 위반",fontsize=9); ax.set_ylim(0,val*1.25)
fig.suptitle("(c) SPD: 단일 최저민감 layer{60} drop조차 게이트 FAIL",fontsize=10)
fig.tight_layout(); fig.savefig(f"{D}/fig_c_spd_gate_fail.png"); plt.close()

# (d) 4-레버 × 벽 매트릭스
fig,ax=plt.subplots(figsize=(6.5,2.8)); ax.axis("off")
rows=["L1 routed-volume↓","L2 sync 제거","L3 precision 압축","L4 overlap(은닉)"]
walls=["(E) routing-invariant\n(allgather)","(Q) 동등 위반\n(sync-drop FAIL)","HW미지원/정확도\n(FP8)","축소 아님\n(prefill-only upstream)"]
status=["❌ E-void","❌ Q-void","❌ HW/acc-void","➖ 비축소"]
cell=[[rows[i],walls[i],status[i]] for i in range(4)]
tb=ax.table(cellText=cell,colLabels=["통신-축소 레버","막히는 벽","판정"],loc="center",cellLoc="left")
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1,1.8)
ax.set_title("(d) 통신-축소 레버의 완전 분류 → 각각 한 벽에 막힘 → win-set=∅ (scoped)",fontsize=9,pad=10)
fig.tight_layout(); fig.savefig(f"{D}/fig_d_taxonomy_matrix.png"); plt.close()

# (e) DP +182% (config 이득, 양성 대조)
fig,ax=plt.subplots(figsize=(4.2,3.2))
b=ax.bar(["TP8\n(통신 50%)","DP8\n(통신 0)"],[5393,15196],color=["#e45756","#54a24b"])
for bar,v in zip(b,[5393,15196]): ax.text(bar.get_x()+bar.get_width()/2,v+200,f"{v}",ha="center")
ax.set_ylabel("aggregate gen_tps"); ax.set_title("(e) 양성 대조: 통신 이득은 config에서\n70B FP4 DP 복제 = +182% (알고리즘 아님)")
ax.annotate("+182%",xy=(1,15196),xytext=(0.3,11000),fontsize=12,fontweight="bold",color="#54a24b")
fig.tight_layout(); fig.savefig(f"{D}/fig_e_dp_config_win.png"); plt.close()

# (f) reduced-expert acceptance 곡선
fig,ax=plt.subplots(figsize=(4.4,3.2))
k=[1,2,8]; a=[0.478,0.804,1.0]
ax.plot(k,a,"o-",color="#4c78a8",ms=8);
for kk,aa in zip(k,a): ax.text(kk,aa+0.02,f"{aa:.3f}",ha="center",fontsize=9)
ax.set_xticks(k); ax.set_xlabel("draft top-k"); ax.set_ylabel("acceptance a (vs top-8 argmax)")
ax.set_title("(f) reduced-expert acceptance (R1-671B)\ntop-2=0.804 OK, 단 FLOPs-only ~1.06× 비신규")
ax.set_ylim(0.4,1.05); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{D}/fig_f_acceptance.png"); plt.close()

print("saved 6 figs to", D)
for f in sorted(os.listdir(D)): print(" ", f)
