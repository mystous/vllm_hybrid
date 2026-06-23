# 적대적 검증 (RED TEAM) — 특성화 논문 자기-공격 기록

*2026-06-17. "학문적 원수" 역할로 실증 매장 시도. 결과를 분칠 없이 기록.*

## 목적
"win-set is empty / 통신-축소 알고리즘 구조적 불가능" 명제를 **실측으로 박살**내려는 적대적 시도.
통과하면 논문 강건성 입증, 깨지면 논문 폐기. 정직하게 어느 쪽이든 기록.

## 공격 ↔ 실측 결과
| # | 공격(급소) | 실험 | 결과 | 판정 |
|---|---|---|---|---|
| 1 | "C3는 기본 allgather strawman; DeepEP면 통신 줄어 L1 부활" | `deep_ep` 설치 시도 | **빌드 실패 — NVSHMEM 부재**(`libnvshmem` 없음, wheel build error) | 킬샷 불발(환경) |
| 1b | "naive 백엔드면 true all-to-all" | TP8+EP8 `--all2all-backend naive` nsys | **naive도 `ncclAllGather_RING` 사용**(allgather 기반 동일, EP 커널 구조 불변) | 역효과 — C3 강화 |
| 1c | "DP+EP는 부팅 가능, 너희가 못한 것뿐" | DP8+EP NCCL 재시도(CUMEM/symm-mem/naive) | NCCL unhandled cuda error(DP7, profile_run all-to-all) 재현 | 차단 지속 |
| 2 | "지배통신 73%는 eager artifact" | (논문 자백) cudagraph=31% | 31%여도 단일 최대 항; L2 동등위반은 config-무관 | 결론 불변 |
| 3 | "게이트가 자의적; lossy 양자화는 통과시키며 sync만 적대시" | (논리) | 유효 지적이나 L2는 단일 layer도 2.6× 초과 → 보정 없인 통과 불가 | 부분 타당, 결과 불변 |
| 4 | "reduced-expert self-spec은 token-exact(Q완벽)+win" | TP+EP 경제성 | draft가 지배 TP all-reduce(73%) full 지불 → net ~1.06×, DP+EP 필요한데 부팅불가 | 약한 win, 강버전 차단 |

## 냉정한 평결
**이 하드웨어/스택(8×B200, NVSHMEM無, vLLM 본 버전)에서 논문의 실증적 핵심 주장은 적대적 공격을 견뎠다.**
- 사용 가능한 **전** all2all 백엔드(default·naive)가 allgather 기반 = routing-invariant(top2/top8 통신 0.95×) **재확인**.
- 유일한 sparse-a2a(DeepEP)는 NVSHMEM 부재로 **설치 불가**. DP+EP는 NCCL로 **부팅 불가**.
- 즉 C3·C5·win-set 논증을 깰 실험들이 전부 defender 쪽으로 떨어짐 → 실증적 매장 **실패**.

## 단, 한 가지 정당한 상처 (인식론 — 실증 아님)
명제의 **"impossibility / win-set is EMPTY"** 는 **과대주장**이다. 정확한 사실은
*"NVSHMEM-less 단일노드 allgather-only 스택에서 win을 찾지 못했다"*.
결정적 반증 regime(DeepEP sparse-a2a, DP-attention+EP)이 **본 셋업에서 실행 불가능**한 것은
*불가능 증명이 아니라 미검증*이다. "셋업이 고장나 X를 못 돌렸다"로 "X 불가능"을 주장 불가.

## 강제 수정 (논문에 반영)
- 제목 격하: ~~"The Empty Win-Set"~~ → **"A Scoped Negative Result: Why Communication-Reducing
  Algorithms Fail on Allgather-Based EP Backends under Strict Distribution-Equivalence — and the
  Untested Sparse-A2A Escape Hatch"**.
- "impossible/empty"→"not found within tested backends/configs". DeepEP·DP+EP는 **open question**으로 명시
  (닫지 않음). 이 셋업 한계(NVSHMEM 부재, DP+EP NCCL 버그)를 Limitations에 1급 제약으로 격상.

## 메타 가치
적대적 검증이 (1) 실증 강건성 입증, (2) 유일 결함(과대주장)을 정밀 지목, (3) 정확한 수정안 산출.
역설적으로 논문을 **더 방어 가능**하게 만듦 — scoped negative result는 overclaim보다 훨씬 견고.
