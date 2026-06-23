# SUB_260 Phase B — 후보 triage + 통합 특성화 통찰

*2026-06-17. candidate#1 B0 NO-GO 후 fallback 후보 점검.*

## 통합 근본 원인 (★ 핵심 특성화 통찰)
vLLM 기본 EP all2all 백엔드 = **`allgather_reducescatter`**: 모든 토큰을 all-gather(전 rank가 전부 수신)
→ 각 rank가 로컬 expert만 계산 → reduce-scatter. 따라서 **inter-GPU 통신량 = (전 토큰 × hidden), 라우팅 결정과
무관**:
- top_k 축소(후보#1): 통신 불변 — 측정 ncclAllGather top-8 vs top-2 = **0.95×** (B0 측정5).
- 복제본 위치(후보#2): 통신 불변 — allgather는 위치 무관, 복제본 선택은 **계산 분배(load)만** 변경.
→ **결론: vLLM 기본 allgather EP에서 "라우팅 측 통신 최적화"는 구조적으로 불가능.** 통신은 routing-invariant.
오직 **DeepEP true-all2all**(token을 선택 expert 보유 rank로만 dispatch)에서만 통신이 routing-dependent.
단 DeepEP 경로는 (a)DP+EP NCCL 부팅블로커 (b)LL slot-padding (c)미검증 3중 조건.

## 후보별 판정
| 후보 | 갭(vLLM 미적용?) | 통신 절감 가능? | 판정 |
|---|---|---|---|
| #1 축소-expert self-spec | ✅ 신규 | ❌ allgather 불변(0.95×) | **NO-GO** (B0) |
| #2 replica-affinity routing | ✅ 신규(`base_router.py:46` random replica `offs%count`, 토폴로지 무관) | ❌ allgather선 load만 바뀜·통신 불변; DeepEP선 가능하나 EPLB load-balancing과 상충 + DP+EP 블로커 | **NO-GO** (동일 allgather 벽) |
| #3 SPD-on-MoE (Sync-Point-Drop) | △ (SPD는 발표됨, MoE EP 적용은 미발표) | **TP all-reduce(73% 지배)를 drop=제거** — allgather와 무관한 다른 통신축 | 🟡 **유일 잔존** — 단 분포동등 게이트 리스크 |

## 핵심 재구성: 지배 통신은 EP all-to-all이 아니라 TP all-reduce
B0 측정3 재해석: R1 TP8+EP8에서 GPU 73% = `cross_device_reduce`(attention/dense **TP all-reduce**),
EP all-to-all(allgather)은 소수. **따라서 통신 공략 타깃은 EP all-to-all이 아니라 TP all-reduce여야 한다.**
- 후보#1/#2는 EP all-to-all(소수+allgather불변)을 노려서 이중으로 빗나감.
- **후보#3 SPD는 TP all-reduce(지배·73%)를 직접 제거** → 타깃이 맞음. SPD = 정확도 저민감 attention 블록의
  sync(all-reduce)를 drop. 70B 8GPU −20% latency/<1% 정확도(논문). **MoE EP 적용은 미발표 = 신규 여지.**
- **리스크**: 본 프로젝트 게이트(max_logprob_diff≤0.5, ppl_rel≤0.1)는 SPD 논문 "<1%"보다 엄격 →
  drop 가능 layer가 좁아질 수 있음. drop layer 선정 calibration 필요.

## 다음
후보#3(SPD-on-MoE)이 유일하게 지배 통신(TP all-reduce)을 타깃하는 생존 후보. 평가 순서:
(1) vLLM에 SPD/sync-drop 유사 기능 있는지 upstream 확인(분칠금지),
(2) 없으면: 어느 layer의 attention all-reduce를 drop해도 분포동등 게이트 통과하는지 오프라인 calibration
   (R1 layer별 sync-drop 민감도 → logprob_diff/ppl_rel 측정),
(3) 통과 layer가 충분하면 PoC(해당 layer all-reduce skip) → tps 향상 측정.
SPD도 막히면 → **특성화 논문 consolidate**(통합 근본원인=allgather routing-invariance + TP all-reduce 지배 +
DeepEP/DP+EP 블로커 + reduced-expert acceptance 곡선; 이미 다수 실측 보유 = publishable).

## #3 SPD 실측 결과 (2026-06-17, faithful 구현) — ❌ NO-GO
구현: `deepseek_v2.py` MLA o_proj(line 982)에 `VLLM_SPD_DROP_LAYERS` ENV-gated `reduce_results=False`
(drop-layer attention all-reduce 생략). 8 TP워커 hook 발동 확인(strawman 아님). R1 TP8 eager.
게이트 측정(teacher-force 132 position, base vs drop):
| drop config | argmax_match | max_logprob_diff (게이트≤0.5) | ppl_rel (≤0.1) | GATE |
|---|---:|---:|---:|---|
| {layer 60} (단일·최후·최저민감 예상) | 0.909 | **1.288** | **0.199** | **FAIL** |
**결정**: 가장 보수적인 단일-마지막-layer drop조차 게이트를 2.6×(logprob)/2×(ppl) 위반. TP8서 o_proj
all-reduce drop = 해당 layer attention의 7/8 손실이라 본질적으로 lossy. 더 많은 layer drop은 악화.
→ **엄격 분포동등 게이트 하에서 drop 가능 layer = 0개. SPD-on-MoE NO-GO.** (SPD 논문 "<1% acc loss"는
본 프로젝트 게이트보다 훨씬 느슨.) 3개 후보 전부 실증 종결 → **특성화 consolidate 확정.**
