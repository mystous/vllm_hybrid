# SUB_227 — Partition-aware harvest 커널 + IE 지표, 2026-06-15

> **판정: positive — IE 가 커널 규칙 간 3.4× 차이 (게이트 ≥2× 충족).**
> 기존 aggressor 모드{basic/nt/cldemote} = "커널 작성법". IE = BW ÷ victim degr%.

## 결과 (victim 0-7 + aggressor 8-23, baseline ns/load=88.4)
| rule | array_mb | aggr BW(GB/s) | victim degr% | **IE** |
|---|---:|---:|---:|---:|
| basic | 32 | 158 | 134.3 | 1.18 |
| nt | 32 | 161 | 206.9 | 0.78 |
| cldemote | 32 | 150 | 80.7 | 1.86 |
| basic | 8 | 134 | 50.8 | 2.65 |
| basic | 128 | 361 | 148.6 | 2.43 |

## 판정
1. **IE 0.78~2.65 = 3.4× spread** → D14 게이트(규칙간 ≥2×) 충족. portfolio
   선택 기준(SUB_228) 전제 성립.
2. **동일 WS(amb32) 커널규칙 서열**: cldemote(1.86) > basic(1.18) > **nt(0.78)**.
   - cldemote: 소비완료 라인 자발 강등 → LLC 오염 최소 = 가장 정중 (가설 일치).
   - **NT-store 의외 최악**: cache bypass 이지만 write-BW 포화가 victim 메모리지연을
     더 키움 (degr 206.9% — basic 의 1.5배). "NT=정중" 통념 반증.
3. **cache-blocking 효과**: 작은 WS(amb8, IE 2.65)=간섭 최소(캐시 적합), 큰 WS
   (amb128, IE 2.43)=높은 BW(361)로 IE 보전. 중간(amb32)이 최악 IE.

## 함의 (SUB_228 입력)
- 정중한 harvest 커널 = **cldemote + cache-blocked(WS≤CAT way) + NT-store 회피**.
- IE 지표가 portfolio 선택 정량 기준으로 유효.

## 비고
- BW 캡처 flaky(aggressor kill 전 print) → basic_32/cldemote_32 BW 단독 보충.
- THP madvise·SW prefetch 거리 규칙은 미시험(추가 A/B 가능).

산출물: `runs/results.csv`. 도구: `src/victim_aggressor.c` (기존 모드).
