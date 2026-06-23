# 특성화 논증 골격 (LOGIC) — Why Communication-Reducing Algorithms Fail in Production MoE Serving under Strict Distribution-Equivalence

*논리 우선. 모든 단계는 본 세션/SUB 실측에 정박. 분칠 금지 — 빈 곳은 빈 곳으로.*

## 0. 명제 (claim)
> 프로덕션 스택(vLLM) + 엄격한 분포-동등 제약(max_logprob_diff ≤ 0.5 AND ppl_rel ≤ 0.1) 하에서,
> 대형 MoE 서빙의 **통신을 줄이는 "알고리즘적" 레버의 win-set은 공집합**이다.
> 통신 이득은 알고리즘이 아니라 **백엔드·병렬화 config 선택**에서만 나온다.

이것은 성능 기법 제안이 아니라 **negative-space 특성화**다. 가치 = "왜 이 분야의 smart-routing/
smart-sync 논문들이 프로덕션·동등제약 하에서 안 통하는가"를 닫힌 논증 + 측정으로 규명.

## 1. win의 정의 (두 조건의 곱)
통신 최적화가 "win"이려면 **동시에**:
- **(E) Effectiveness**: 지배 통신 비용을 실제로 줄인다.
- **(Q) Equivalence**: 출력 분포를 게이트 내로 보존한다.
(novelty는 직교 — 논증은 E×Q에 관한 것. 어떤 레버는 E도 Q도 못 만족.)

## 2. 측정 신뢰성 전제 (없으면 타깃을 틀림)
**Lemma 1 (방법론, C1)**: 통신 비중은 CPU 스레드 점유(flamegraph)가 아니라 GPU 커널 타임라인(nsys)으로
측정해야 한다. 실측: 동일 all-reduce가 flamegraph 6% vs nsys GPU 31~73%. (cudagraph가 launch를 GPU로
내려 CPU엔 안 보임.) → 이후 모든 "지배 통신" 주장은 nsys 기반.

## 3. 지배 통신의 위치 (타깃 재설정)
**Lemma 2 (C2)**: TP8+EP8 R1에서 지배 통신은 expert all-to-all이 아니라 **attention/dense의 TP
all-reduce**다. 실측(nsys, eager): `cross_device_reduce` = GPU 72.9%, EP dispatch/combine = ~1%.
(cudagraph서 ~31%로 완화되나 여전히 단일 최대 항.) → MoE라고 expert 통신을 노리면 빗나간다.

## 4. 통신-축소 레버의 완전 분류 (exhaustive taxonomy)
inter-GPU 통신은 두 소스뿐: **(i) TP all-reduce**(activation 합산), **(ii) EP all-to-all**(token dispatch).
통신을 "줄이는" 방법은 물리적으로 네 가지뿐:
| # | 레버 | 무엇을 함 | 대상 | 이 세션 후보 |
|---|---|---|---|---|
| L1 | **routed-volume 축소** | 보내는 (token×expert) 줄임 | (ii) | #1 축소-expert, #2 replica-affinity |
| L2 | **sync 제거** | all-reduce 자체를 건너뜀 | (i) | #3 SPD |
| L3 | **payload 정밀도 압축** | bf16→FP8 등 비트폭↓ | (i),(ii) | (이전 세션) FP8-AR / FP8-residual |
| L4 | **overlap (은닉)** | 통신을 연산 뒤에 숨김(총량 불변) | (i),(ii) | async-TP/TokenWeave 등 |
L4는 *축소가 아니라 은닉* → 정의상 통신량 불변이고 prefill-only가 이미 upstream(SUB_256). 따라서
**진짜 "축소" 레버는 L1·L2·L3 셋뿐.** 이 셋을 각각 죽이면 win-set 공집합 증명 완성.

## 5. 각 레버의 구조적 사망 (핵심, 일대일 대응)
### L1 (routed-volume 축소) → **(E) 위반 — routing-invariant 벽**
**Lemma 3 (C3, ★)**: vLLM 기본 EP all2all 백엔드 `allgather_reducescatter`는 **모든 토큰을 all-gather**한 뒤
각 rank가 로컬 expert만 계산 → inter-GPU 통신량 = (전 토큰 × hidden), **라우팅 결정과 독립**.
- 실측: top-8 vs top-2 강제 → `ncclAllGather` = **0.95×**(불변). 줄어든 건 local permute(compute)뿐.
- 따라서 #1(top_k↓)·#2(replica 위치)는 (ii) 통신을 **원리적으로** 못 줄임. (게다가 (ii)는 §3에서 minor.)
- **유일 탈출구**: DeepEP true-all2all(token을 선택 expert 보유 rank로만 dispatch → routing-dependent).
  그러나 이는 (a) 기본 아님, (b) DP+EP가 NCCL "unhandled cuda error"로 부팅 불가(실측), (c) LL
  slot-padding이 비례 깰 수 있음 — **알고리즘이 아니라 백엔드 교체 문제이고 현재 막힘.**

### L2 (sync 제거) → **(Q) 위반 — equivalence 벽**
**Lemma 4 (C5, SPD faithful 실측)**: TP all-reduce를 drop하면 cross-rank attention 합산을 잃는다.
- 실측: R1 TP8서 **단일·최후·최저민감 layer{60}** 의 o_proj all-reduce drop만으로도
  max_logprob_diff **1.288**(게이트 0.5의 2.6×), ppl_rel **0.199**(0.1의 2×) → **FAIL**.
- TP8서 o_proj AR drop = 그 layer attention의 7/8 손실(본질적 lossy). 더/이른 layer는 악화.
- → 게이트 하 **drop 가능 layer = 0개.** L2는 (E)는 만족하나 (Q)를 못 만족.

### L3 (정밀도 압축) → **(E) HW-blocked / (Q) accuracy 붕괴** (이전 세션, 인용)
- FP8 all-reduce: B200 multimem.ld_reduce가 FP8 reduce **HW 미지원**, naive 2-shot 6.4× 느림 → (E) 不.
- FP8 residual stream: 61층 누적 logit_diff 1.9 → (Q) 不.

## 6. 결론 (the theorem)
L1·L2·L3가 각각 정확히 한 벽에 막힌다:
- **routing-preserving 레버(L1)** = (E)-void: 기본 백엔드선 통신이 routing-invariant.
- **comm-eliminating 레버(L2)** = (Q)-void: sync-drop은 분포 동등 위반.
- **precision 레버(L3)** = HW-/accuracy-void.
두 조건 E와 Q를 동시에 만족하는 레버가 없음 → **알고리즘적 통신-축소 win-set = ∅.** ∎

## 7. So-what (reframing) — 양성 대조로 논증 완결
통신 이득이 **존재하긴 한다 — 단 알고리즘이 아니라 config에서**:
- **fits-on-GPU → DP 복제**가 (i) TP all-reduce를 통째 제거: 70B FP4 DP8 **+182%**(SR-005, 실측).
  이는 라우팅·sync 변경이 아니라 **병렬화 config 선택**.
- 남은 (ii)-축소 경로(DeepEP)도 **백엔드 엔지니어링** 문제(부팅 버그가 막음), 알고리즘 아님.
→ 기여 명제: "smart routing/sync saves comm"은 잘못된 계층을 본다. 프로덕션·동등제약 하 통신 레버는
**백엔드 + 병렬화 선택**이다. 알고리즘 설계 공간은 비어 있다.

## 8. 예상 반론 ↔ 방어 (논증 견고화)
- "DeepEP면 L1 통신 준다" → 맞다. 그래서 **알고리즘이 아니라 백엔드 문제**라는 게 명제. 게다가 DP+EP
  부팅 블로커로 현재 실현 불가(실측). DeepEP가 못 한다고 주장하지 않음 — win이 비알고리즘적이라고 주장.
- "73%는 eager artifact, cudagraph는 31%" → 31%여도 단일 최대 항이고, L2(sync-drop) 동등위반은
  config 무관(수학적). 논증 불변.
- "SPD를 layer 하나만 봤다" → **최저민감(최후) layer**를 봤고 이미 2.6× 위반. 더/이른 layer는 단조 악화 →
  best-case가 실패 = 전구간 실패.
- "게이트가 과도하게 엄격하다" → 본 프로젝트 제약(분포 동등). **느슨한 acc(SPD의 <1%)면 L2 통과 가능** —
  이 의존성 자체가 finding(§9): 동등 정의가 feasibility를 가른다.

## 9. Scope / limits (정직)
- 단일 노드 8×B200, vLLM 특정 버전, R1-671B/K-EXAONE 일반화. 멀티노드·타 프레임워크 미검증.
- 결론은 **엄격 분포-동등 게이트에 조건부**. 게이트를 정확도-허용으로 완화하면 L2·L3 일부 부활 가능 —
  이는 한계가 아니라 축(axis): "동등 정의 → 통신 레버 feasibility" 맵이 본 특성화의 한 기여.
- DeepEP true-all2all 경로는 **닫지 않음**(부팅 블로커로 미측정) — 유일한 열린 비알고리즘 탈출구로 명시.

## 10. 논문 매핑 (다음 writeup 시)
§0→Abstract/Thesis, §2-3→측정방법·병목(C1/C2), §4→taxonomy(독창 프레이밍), §5→core results(C3/C5+L3인용),
§6→discussion(SR-005 대조), §8→rebuttal, §9→limitations. 그림: (a)flamegraph vs nsys, (b)top-k vs allgather
0.95×, (c)SPD layer{60} 게이트 위반 막대, (d)DP+EP 부팅 블로커 표, (e)taxonomy 4×벽 매트릭스.
