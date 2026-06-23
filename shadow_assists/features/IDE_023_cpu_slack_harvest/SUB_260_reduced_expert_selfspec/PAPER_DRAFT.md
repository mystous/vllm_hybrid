# A Scoped Negative Result: Why Communication-Reducing Algorithms Fail on Allgather-Based EP Backends under Strict Distribution-Equivalence — and the Untested Sparse-A2A Escape Hatch

*내부 draft, 2026-06-17. 모든 수치 = 본 세션 실측(8×B200, vLLM v1.7.dev16186). 측정 연구(systems characterization).*

> **⚠ Scope (적대적 검증 RED_TEAM.md 반영)**: 본 결과는 **NVSHMEM 부재·단일노드·allgather-only** 스택에
> 한정된다. "win-set is empty"는 **검증된 백엔드/config 내에서**의 의미이며 universal impossibility가 아니다.
> 결정적 반증 regime — **DeepEP sparse-all2all**(NVSHMEM 부재로 설치 불가)과 **DP-attention+EP**(NCCL
> 부팅 버그) — 는 **미검증 open question**으로 명시한다. 본 셋업에서 이들이 실행 불가능한 것은 불가능
> 증명이 아니다.

---

## Abstract
대형 MoE(DeepSeek-R1 671B, K-EXAONE-236B)를 8×NVIDIA B200에서 vLLM으로 서빙할 때 통신이 지배
병목임은 알려져 있고, 이를 줄이려는 "더 똑똑한 라우팅/동기화" 기법이 다수 제안돼 왔다. 본 연구는 이들이
**프로덕션 스택 + 엄격한 출력 분포-동등 제약**(per-token logprob max-abs-diff ≤ 0.5, 시퀀스 PPL
relative-diff ≤ 0.1) 하에서 **구조적으로 실패함**을 측정으로 보인다. 기여: (1) 통신 비중은 flamegraph가
아닌 nsys GPU 타임라인으로 측정해야 함을 정량화(6% vs 31–73%); (2) 지배 통신이 expert all-to-all이 아니라
attention/dense의 **TP all-reduce**임을 규명; (3) 통신-축소 레버의 **완전 분류**(routed-volume / sync-제거 /
precision-압축 / overlap)와, 각 레버가 어느 구조적 벽에 막히는지의 일대일 대응 — 특히 vLLM 기본
all-to-all(allgather)이 **routing-invariant**(top-2 vs top-8 통신 0.95×)임과, sync-drop이 단일 최저민감
layer에서도 게이트를 2.6× 위반함을 실측; (4) 따라서 알고리즘적 통신-축소 win-set이 공집합이며, 실재하는
통신 이득(fits-on-GPU → DP 복제 +182%)은 알고리즘이 아니라 **백엔드·병렬화 config 선택**에서 나옴을
보인다. 결론: 통신 최적화는 잘못된 추상화 계층(알고리즘)에서 탐색돼 왔다.

## 1. Introduction
MoE는 추론 시 expert 분산(EP)을 요구하고, EP는 all-to-all 통신을 낳는다. 그래서 "통신을 줄이는 라우팅"
(reduced-expert speculation, affinity routing) 과 "동기화를 줄이는" (sync-point drop) 기법 군이 활발하다.
질문: **이들이 실제 프로덕션 스택에서, 그리고 출력이 원본과 동등해야 한다는 제약 하에서 통하는가?**
본 연구는 8×B200·vLLM·R1-671B에서 직접 측정해, 답이 **아니오**이며 그 이유가 우연이 아니라 **구조적**임을
보인다. 핵심 통찰: 통신 최적화의 진짜 레버는 알고리즘이 아니라 백엔드(all-to-all 구현)와 병렬화 config다.

## 2. Background & Methodology
- **병렬화**: attention/dense는 TP(rank마다 head 분할 → o_proj 후 **all-reduce**로 합산), MoE는 EP
  (expert 분산 → **all-to-all** dispatch/combine). R1은 TP8+EP8 또는 DP-attention+EP.
- **all-to-all 백엔드**: vLLM 기본 `allgather_reducescatter`(전 토큰 all-gather→로컬 expert→reduce-scatter)
  vs `DeepEP`(token을 선택 expert 보유 rank로만 dispatch). 둘의 통신 특성이 본 연구의 핵심 분기.
- **동등 게이트(binding)**: max_logprob_diff ≤ 0.5 AND ppl_rel ≤ 0.1. (token-level 일치는 정보용.)
- **측정**: nsys `cuda_gpu_kern_sum`(GPU 커널 타임라인). teacher-force `prompt_logprobs`로 게이트 지표 산출.
- **Max-config baseline**(SUB_259): 모든 성능 플래그 ON(O3 FaP cudagraph + flashinfer-trtllm AR +
  deep_gemm + symm_mem) R1 EP8 = c64 **2,365 gen_tps**. (함정: `VLLM_USE_FLASHINFER_MOE_FP8`은 R1
  block-FP8 미지원 BOOT_FAIL.)

## 3. Where is the communication? (기여 1·2)
**3.1 flamegraph ≠ GPU 병목 (C1)**: 동일 all-reduce가 CPU 스레드 점유(py-spy) **6%**, GPU 커널시간(nsys)
**31–73%**. cudagraph가 launch를 GPU로 내려 CPU 프로파일에 안 보임. → 통신 비중은 반드시 nsys로.

**3.2 지배 통신은 TP all-reduce (C2)**: R1 TP8+EP8 eager nsys —
| 커널 | GPU 시간% |
|---|---:|
| `cross_device_reduce_2stage` (TP all-reduce) | **72.9** |
| deep_gemm fp8 (expert+dense GEMM) | ~12 |
| EP dispatch/combine (`ep_scatter/gather`+allgather) | **~1** |
직관과 반대: MoE인데 통신 병목은 expert 통신이 아니라 attention의 TP all-reduce. (cudagraph서 ~31%로
완화되나 단일 최대 항.)

## 4. A Complete Taxonomy of Communication-Reduction Levers (기여 3, 프레이밍)
inter-GPU 통신 소스는 둘뿐: (i) TP all-reduce, (ii) EP all-to-all. 통신을 *줄이는* 방법은 물리적으로 4가지:
| 레버 | 작용 | 대상 | 본 연구 후보 |
|---|---|---|---|
| **L1 routed-volume↓** | 보내는 token×expert 감소 | (ii) | 축소-expert spec, replica-affinity |
| **L2 sync 제거** | all-reduce 건너뜀 | (i) | SPD (sync-point drop) |
| **L3 precision 압축** | bf16→FP8 등 | (i),(ii) | FP8 all-reduce / FP8 residual |
| **L4 overlap** | 통신을 연산 뒤 은닉(총량 불변) | (i),(ii) | async-TP, TokenWeave |
L4는 *축소가 아닌 은닉*(총량 불변)이고 prefill-only가 이미 upstream → 본 분석 제외. **진짜 축소 = L1·L2·L3.**

## 5. Why Each Lever Fails (core results)
**5.1 L1 → Effectiveness-void (routing-invariance, C3)**
vLLM 기본 `allgather_reducescatter`는 **모든 토큰을 all-gather** → 통신량 = 전 토큰 × hidden, **라우팅 독립**.
- 실측: top-8 vs top-2 강제 → `ncclAllGather` **0.95×**(불변). 줄어든 건 local permute(`ep_gather` 0.59×,
  `ep_scatter` 0.49× = compute)뿐.
- 결과: 축소-expert(top_k↓)·replica-affinity(위치↓) 모두 (ii) 통신을 원리적으로 못 줄임. 게다가 (ii)는 §3.2서 minor.
- (부수) reduced-expert acceptance 곡선: R1 top-1 a=0.478, top-2 a=0.804 — FLOPs-only 속도배율 ~1.06× = 비신규.

**5.2 L2 → Equivalence-void (sync-drop, C5)**
SPD를 faithful 구현(`deepseek_v2.py` MLA o_proj `reduce_results=False`, ENV `VLLM_SPD_DROP_LAYERS`,
8 TP워커 발동 확인). 측정:
| drop | argmax_match | max_logprob_diff (≤0.5) | ppl_rel (≤0.1) | GATE |
|---|---:|---:|---:|---|
| {layer 60} 단일·최후·최저민감 | 0.909 | **1.288** | **0.199** | **FAIL** |
단일·최저민감 layer drop조차 게이트 2.6×(logprob)/2×(ppl) 위반. TP8서 o_proj AR drop = 그 layer attention의
7/8 손실(본질적 lossy). 더/이른 layer는 단조 악화 → **게이트 하 drop 가능 layer 0개.**

**5.3 L3 → HW/accuracy-void (precision, 이전 세션 인용)**
FP8 all-reduce: B200 multimem.ld_reduce가 FP8 reduce HW 미지원, naive 2-shot 6.4× 느림 → E 不.
FP8 residual stream: 61층 누적 logit_diff 1.9 → Q 不.

## 6. The Win-Set is Empty — within tested backends/configs (결론, scoped)
L1(E-void) ∧ L2(Q-void) ∧ L3(HW/acc-void) ∧ L4(축소 아님). **검증된 스택(allgather EP, TP+EP, 엄격
게이트)** 내에서 E와 Q를 동시에 만족하는 알고리즘적 통신-축소 레버가 존재하지 않는다 → win-set = ∅
*(scoped)*. **Open**: DeepEP sparse-a2a(L1의 escape, NVSHMEM 부재로 미검증)·DP-attention+EP(부팅 버그로
미검증)에서는 이 결론이 성립하지 않을 수 있다 — 적대적 검증(RED_TEAM.md)이 정확히 이 지점을 지목했고,
본 셋업에서 닫지 못했다. 따라서 본 논문은 **impossibility가 아니라 scoped negative result + open escape**.

## 7. Where Gains Actually Come From (양성 대조)
통신 이득은 실재한다 — 단 알고리즘이 아니라 **config/백엔드**에서:
- **config**: fits-on-GPU(70B FP4 40GB) → **DP 복제(DP8)** 가 TP all-reduce를 통째 제거 → **+182%**
  (5,393→15,196 gen_tps, SR-005). 라우팅·sync 변경이 아니라 병렬화 선택.
- **backend**: (ii)를 routing-dependent로 만드는 유일 길 = DeepEP true-all2all. 그러나 (a) 기본 아님,
  (b) DP+EP가 NCCL "unhandled cuda error"로 부팅 불가(실측), (c) LL slot-padding 우려 — **백엔드
  엔지니어링 문제이고 현재 막힘.** 알고리즘 아님.
→ 명제: "smart routing/sync saves comm"은 잘못된 추상화 계층을 본다.

## 8. Discussion / Rebuttals
- "DeepEP면 L1 통신 준다" → 맞다. 그래서 win이 **알고리즘이 아니라 백엔드**라는 게 본 명제. DeepEP 무능
  주장 아님. 게다가 DP+EP 부팅 블로커로 현재 실현 불가(실측).
- "73%는 eager artifact, cudagraph 31%" → 31%여도 단일 최대 항. L2 동등위반은 config-무관(수학적). 불변.
- "SPD를 한 layer만" → **최저민감(최후)** layer가 이미 2.6× 위반 = best-case 실패 → 전구간 실패.
- "게이트가 과하다" → 본 제약은 분포-동등. 느슨한 acc(SPD <1%)면 L2 통과 가능 — **이 의존성 자체가 finding**:
  동등 정의가 feasibility를 가른다(§10).

## 9. Related Work (대비표)
| 기법 | 노린 것 | 본 연구 매핑 | 프로덕션·동등 하 결론 |
|---|---|---|---|
| SS-MoE / MoE-Spec (reduced-expert) | (ii) FLOPs↓ | L1, 단일디바이스 | allgather서 통신 무이득, FLOPs ~1.06× 비신규 |
| Speculative MoE (2503.04398) | EP all-to-all pre-schedule | L1 변종(draft 아님) | routing-invariant 벽 동일 |
| GRACE-MoE (affinity routing) | (ii) locality | L1(#2) | allgather서 load만 변경, 통신 불변 |
| SPD / Sync-Point-Drop (ICML'25) | (i) sync 제거 | L2 | 엄격게이트서 단일 layer도 FAIL |
| TokenWeave / Flux / Comet | overlap | L4(은닉) | 총량 불변, prefill-only upstream |
| Flash-Comm / low-bit AR | (i),(ii) precision | L3 | B200 FP8 HW 미지원 / accuracy 붕괴 |

## 10. Limitations / Scope
- 단일 노드 8×B200, vLLM 특정 버전, R1/K-EXAONE. 멀티노드·타 프레임워크 미검증.
- 결론은 **엄격 분포-동등 게이트에 조건부** — 정확도-허용으로 완화하면 L2·L3 일부 부활 가능. 이는 한계가
  아니라 축: "동등 정의 → 통신 레버 feasibility" 맵이 본 연구의 한 기여.
- DeepEP true-all2all 경로는 **닫지 않음**(부팅 블로커로 미측정) — 유일한 열린 비알고리즘 탈출구로 명시.

## Figures (생성 완료 — `runs/figs/`, 영문 라벨, 모두 실측값)
- (a) `fig_a_flamegraph_vs_nsys.png` — flamegraph 6% vs nsys cudagraph 31% vs eager 73% (방법론).
- (b) `fig_b_routing_invariant.png` — top-8 vs top-2: ncclAllGather 0.95×(불변) vs local permute 0.5×(compute).
- (c) `fig_c_spd_gate_fail.png` — SPD layer{60}: logprob 1.288(2.6× over 0.5), ppl_rel 0.199(2× over 0.1).
- (d) `fig_d_taxonomy_matrix.png` — L1–L4 × 벽 매트릭스 → win-set ∅ (scoped).
- (e) `fig_e_dp_config_win.png` — DP8 vs TP8 +182% (config 이득, 양성 대조).
- (f) `fig_f_acceptance.png` — reduced-expert acceptance top1=0.478/top2=0.804/top8=1.0.
생성 스크립트: `make_figs.py`(데이터 인라인=BASELINE/B0_FINDINGS/runs nsys 표 출처).

## Reproducibility (artifacts)
- nsys: `SUB_257/runs/r1cg_prof.nsys-rep`, `SUB_260/runs/comm_top{8,2}.nsys-rep`.
- 측정: `SUB_259/BASELINE.md`, `SUB_260/{B0_FINDINGS,CANDIDATE_TRIAGE,LOGIC}.md`, `accept_bench.py`,
  `spd_gate.py`, `nsys_comm*.sh`.
- 계측 hook(ENV-gated, no-op 기본): `base_router.py`(`VLLM_MOE_FORCE_TOPK`), `deepseek_v2.py`(`VLLM_SPD_DROP_LAYERS`).
- 관련 success: SR-001~006.
