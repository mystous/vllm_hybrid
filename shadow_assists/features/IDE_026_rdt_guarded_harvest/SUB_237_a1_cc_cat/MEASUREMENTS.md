# SUB_237 — CC-CAT (AIMD elastic CAT), 2026-06-15

> **판정: 기각 — actuator(CAT) 무효.** CAT way 제어가 BW 간섭에 권한이 없어
> AIMD 제어 루프가 open-loop. (SUB_221 과 동일 근본 원인.)

## 정적 CAT 곡선 (선행 — AIMD 가 추종할 대상)
캐시상주 aggressor(4스레드×4MB=16MB) + victim, harvest CLOS CAT mask sweep:
| harvest ways | victim ns/load |
|---|---|
| none(무harvest) | 84.7 |
| 20-way (fffff) | 92.9 |
| 8-way (000ff) | 94.6 |
| 4-way (0000f) | 90.8 |
| 2-way (00003) | 92.0 |

→ **CAT ways 2~20 전부 victim ~92 (무효, 노이즈 내)**. 곡선이 평평 = AIMD 가 추종할
   기울기 없음.

## 판정
1. **CAT actuator 가 victim SLO 에 권한 없음**: 캐시상주 aggressor(약 +9%)는 CAT 변화
   무관, SUB_221 의 강-aggressor(+140%)도 CAT 무관. 즉 LLC-occupancy 스펙트럼 전체에서
   CAT 무효.
2. **근본 원인**: CAT 은 LLC *점유*(ways)를 제어하나 간섭 기제는 메모리 *BW*. 캐시상주면
   BW 적어 간섭 자체가 작고(CAT 무의미), BW-streaming 이면 CAT 가 그 BW 를 못 줄임.
   → **CAT 은 메모리 BW 간섭의 actuator 가 될 수 없음**.
3. CC-CAT 의 AIMD(혼잡제어 이식) 설계는 정교하나, **actuator(CAT)가 무효라 제어
   루프가 닫히지 않음** → 알고리즘 기각. 실효 actuator = MBA/SW-MBA(BW) (SUB_224).

## 비고
- idle_inject(intel_powerclamp) 커널스레드가 harvest 코어에 보임 — aggressor 약화 교란
  가능성 있으나, SUB_221 강-aggressor 결과로 CAT 무효는 독립 확정.
- AIMD 골격은 BW actuator(SW-MBA)에 얹으면 유효할 수 있음 (CC-CAT→CC-MBA 변형은 후속).

산출물: `runs/static_cat.csv`.
