# SUB_215 [D2] — MEASUREMENTS (확정판, 2026-06-12)

> **판정 요약 (negative, 범위 한정)**: 스트리밍형 sibling harvest 에 대해 **L2 CAT
> 추가 효과 없음** — S2(12-4)/S3(14-2) 모두 S1(MBA20 단독) 과 noise band (±4~6pp)
> 내 동일. **MBA 단독으로 충분** 이 확정되며, SUB_214 의 "binding 간섭 = 메모리 BW"
> 결론이 L2 계층에서도 확증됨. 단 본 판정은 **스트리밍 aggressor (working set
> 32MB/thread ≫ L2)** 에 한정 — L2-상주형 harvest (branchy/tree) 는 별도 검증 필요
> (SUB_218 AMX·SUB_233 pointer-chase 에서 후속).

## 1. 측정 환경

SUB_214 와 동일 (70B suffix K=32, FaP, conc=32, 7 corpora, vLLM taskset `0-47,56-103`,
**셀별 fresh boot**). 신규 도구: `src/set_l2_mask.py` (harvest CLOS L2 CBM 을 전 112
도메인 일괄 기록 — 실검증 통과).

| 셀 | aggressor (sibling `112-119,168-175`) | MBA | harvest L2 CBM |
|---|---|---|---|
| S0_base | 없음 | — | — |
| S1_mba20 | 16T AVX-512 triad | 20% | ffff (전체 16-way) |
| S2_l2_12_4 | 〃 | 20% | **000f (4-way, 512KB)** |
| S3_l2_14_2 | 〃 | 20% | **0003 (2-way, 256KB)** |

## 2. 결과 (28셀, accept 0.721~0.734 일치)

| corpus | S0 | S1 | S2 | S3 | S1/S0 | S2/S0 | S3/S0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sharegpt | 4,699 | 4,533 | 4,494 | 4,346 | 0.965 | 0.956 | 0.925 |
| swebench | 5,192 | 4,757 | 5,351 | 4,977 | 0.916 | 1.031 | 0.959 |
| humaneval | 4,664 | 4,681 | 4,421 | 4,539 | 1.004 | 0.948 | 0.973 |
| mbpp | 2,583 | 2,552 | 2,502 | 2,512 | 0.988 | 0.969 | 0.973 |
| wildchat | 5,287 | 4,987 | 4,908 | 4,980 | 0.943 | 0.928 | 0.942 |
| lmsys | 3,871 | 4,247 | 4,026 | 3,628 | 1.097 | 1.040 | 0.937 |
| mix | 7,074 | 6,373 | 7,062 | 6,524 | 0.901 | 0.998 | 0.922 |
| **기하평균** | | | | | **0.972** | **0.981** | **0.947** |

harvest 측 (MBM): S1 16.2 / S2 15.8 / S3 15.9 GB/s — L2 mask 가 스트리밍 BW 에 무영향
(예상대로: L2 미스율이 이미 ~100% 인 스트리밍은 L2 way 를 줄여도 잃을 게 없음).
LLC 점유 243→254→267 MB (L2 CAT 는 LLC 점유와 무관 — 정상).

## 3. 판정

1. **D2 게이트 불충족** — S2 가 S1 대비 +0.9pp, S3 는 −2.5pp: 둘 다 baseline 재현
   noise (S0 가 SUB_214 C0 대비 ±4~6% 변동) 안. corpus 단위 일관 패턴 없음
   (swebench S2 +11.5pp ↔ sharegpt/lmsys 역방향 = noise).
2. **설계 규칙 확정**: 스트리밍형 harvest 의 sibling co-location 은 **MBA 가드만으로
   충분, L2 CAT 불요** (CLOS 8개 제한 자원을 아낄 수 있음).
3. **SUB_214 anchor 재현**: S1/S0 = 0.972 (전일 C3/C0 = 0.990) — noise band 내.
   "sibling+MBA20 ≈ serving 손실 ~1-3%" 이틀 연속 성립.
4. **유보 사항**: L2-상주형 (branchy, pointer-chase, tile-resident) aggressor 는 미검증
   — L2 CAT 의 가치는 그쪽에 있을 수 있음. humaneval worst-case 가설 (SUB_214 의
   −9.3%) 은 본 측정에서 재현 안 됨 (S1 humaneval 1.004) → run noise 로 결론.

## 4. 산출물

`runs/summ_*.json` (28셀) / `runs/mon_S{1,2,3}*.csv` / `run_sub215.sh` /
`../src/set_l2_mask.py`

## 5. 후속 연결

- SUB_217 (D4 cldemote): aggressor 변형 비교 — 동일 프로토콜.
- SUB_218 (D5 AMX aggressor) 에서 L2-상주형으로 본 판정의 유보 사항 해소 가능.
- T1.5/governor: "L2 CAT 액추에이터는 스트리밍형엔 무효" — CLOSPACK (SUB_241) 의
  클러스터→자원 매핑 규칙에 반영 (스트리밍형 클러스터에 L2 way 배분 금지).
