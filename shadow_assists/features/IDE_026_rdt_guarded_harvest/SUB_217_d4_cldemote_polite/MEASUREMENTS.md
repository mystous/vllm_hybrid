# SUB_217 [D4] — MEASUREMENTS (확정판, 2026-06-12)

> **판정 요약 (positive + anti-pattern 동시 발견)**:
> ① **cldemote = 첫 커널-작성법 lever 성공** — 무가드 sibling harvest 의 serving 적자
> −8.2% 를 **−3.4% 로 59% 회복** (HW 노브 0개, intrinsic 한 줄). IE 2.1×.
> ② **NT-store = BW-지배 환경의 anti-pattern** — RFO 제거로 harvest BW 77→110 GB/s
> 폭증 → serving −20.3% (basic 보다 2.5× 악화). "캐시 예의 ≠ 버스 예의".

## 1. 측정 환경

SUB_214/215 와 동일 (70B suffix K=32, FaP, conc=32, 7 corpora, 셀별 fresh boot).
aggressor = sibling `112-119,168-175` 16T, **MBA 무가드 (100%)** — 작성법 차이가
드러나도록 의도적 무가드. 구현: `victim_aggressor.c` `--aggr-mode basic|cldemote|nt`
(cldemote: store 직후 A/B/C 라인 `_cldemote`, nt: `_mm512_stream_pd`+sfence).

## 2. 결과 (28셀)

| corpus | A0 | A1 basic | A2 cldemote | A3 nt | A1/A0 | A2/A0 | A3/A0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sharegpt | 4,583 | 4,079 | 4,457 | 3,729 | 0.890 | 0.972 | 0.814 |
| swebench | 5,187 | 4,683 | 5,199 | 4,175 | 0.903 | 1.002 | 0.805 |
| humaneval | 4,532 | 4,011 | 4,309 | 3,751 | 0.885 | 0.951 | 0.828 |
| mbpp | 2,643 | 2,727 | 2,436 | 2,025 | 1.032 | 0.921 | 0.766 |
| wildchat | 5,028 | 4,364 | 4,904 | 3,984 | 0.868 | 0.975 | 0.792 |
| lmsys | 4,367 | 3,867 | 4,037 | 3,066 | 0.885 | 0.924 | 0.702 |
| mix | 6,397 | 6,230 | 6,534 | 5,663 | 0.974 | 1.021 | 0.885 |
| **기하평균** | | | | | **0.918** | **0.966** | **0.797** |

harvest 측 (MBM) + **IE (간섭 효율) 첫 실측**:

| 변형 | harvest BW | LLC 점유 | serving 손실 | **IE = GB/s ÷ 손실pp** |
|---|---:|---:|---:|---:|
| A1 basic | 77.0 GB/s | 276 MB | 8.2pp | 9.4 |
| **A2 cldemote** | 67.8 GB/s | **218 MB (−21%)** | 3.4pp | **19.9 (2.1×)** |
| A3 nt | 109.9 GB/s | 279 MB | 20.3pp | 5.4 (최악) |

## 3. 판정

1. **D4 게이트 통과** — cldemote 변형이 무가드에서 victim 영향을 유의 감소
   (−8.2→−3.4pp, 7 corpus 중 6 방향 일치). LLC 점유 −21% 가 메커니즘 직접 증거.
   비용: harvest BW −12% (smoke 의 −65% 와 달리 실전 16T 는 BW-bound 라 지연 은닉).
2. **NT-store 반전**: LLC 는 안 더럽히지만 (점유 불변은 B/C read 스트림 몫) RFO
   제거로 버스 압력 +43% → BW-지배 간섭에서 역효과. **NT 는 반드시 BW 상한
   (MBA/token-bucket) 과 병행** — 단독 사용 금지 규칙.
3. **IE 서열 확정 (스트리밍 계열)**: cldemote(19.9) > basic(9.4) > nt(5.4)
   — D14 (SUB_227) IE 지표의 첫 실데이터, D15 portfolio 선택 기준 가동.

## 4. 산출물 / 후속

`runs/summ_*.json` 28셀, `runs/mon_A*.csv`, `run_sub217.sh`, `victim_aggressor.c`
(3변형). 후속: ① cldemote+MBA20 조합 (적자 0 수렴 후보) ② nt+MBA20 (LLC-청정 +
BW-상한 동시 — 이론상 최선) — SUB_227 IE 매트릭스에서 조합 셀로. ③ SUB_218 AMX
aggressor 로 L2-상주형 IE 측정.
