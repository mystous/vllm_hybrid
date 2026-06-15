# SUB_241 — CLOSPACK (480-RMID 측정주도 15-CLOS 패킹), 2026-06-15

> **판정: positive — 측정주도 차등 MBA 패킹 작동.** CAT 무효라 CLOS의 MBA가 실효 레버.

## 전제 (실측)
MBA gran=10%, num_closids=15, RMID=480. → 480 RMID로 워크로드 미세 측정, 15 CLOS로
제어. CAT 무효(SUB_221/237)라 CLOS 제어 효과는 MBA.

## 데모 (2 harvest: heavy 8T×32MB / light 4T×4MB)
| arm | victim ns/load | |
|---|---:|---|
| baseline | 82.5 | — |
| naive (둘 다 MBA100) | 136.8 | +66% |
| CLOSPACK (heavy→MBA20, light→100) | 114.3 | +39% (naive −16%) |

## 판정
- **측정으로 BW-heavy 식별 → 그 CLOS만 MBA throttle, light 보존** = naive 대비 victim
  16% 더 보호 + light harvest 풀 유지. **측정주도 차등 제어** 입증.
- 전체 알고리즘(N≫15 워크로드 → 15 CLOS bin-packing)은 480 RMID 측정 + MBA CLOS로
  스케일(전제 검증). 본 데모는 핵심(measure→offender throttle) 확인.

## 비고
- victim 완전보호 아님(+39%): light+잔여 BW 기여. light도 측정해 차등 throttle하면 개선.
- 실효 레버 = MBA(SUB_220/238 정합). CAT 기반 패킹은 무효.

산출물: `runs/results.csv`.
