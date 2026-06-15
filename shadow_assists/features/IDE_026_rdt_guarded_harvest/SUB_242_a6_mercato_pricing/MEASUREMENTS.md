# SUB_242 — MERCATO (혼잡가격 BW 시장), 2026-06-15

> **판정: positive — 작동.** governor 가 혼잡가격 p←p·exp(γ·err) 로 harvest MBA% 를
> 동적조정해 총 BW 를 target 으로 수렴, victim 보호 (CSMA 동급, 연속 proportional 제어).

## 구현
`mercato_ctl.py` (sudo): MBM 으로 총 BW 센싱(100ms) → err=(BW−target)/target →
p=clamp(p·exp(γ·err),1,10) → harvest CLOS MBA% = 100/p (10% grain). resctrl schemata
write 는 root 권한 필요 → sudo 실행 (초기 user-write 권한거부 버그 수정).

## 결과 (victim 0-7, harvest 8-23, target 5000 MB/s, γ=0.5)
| arm | victim ns/load | |
|---|---:|---|
| baseline | 86.9 | — |
| unthrottled | 206.1 | +137% |
| **MERCATO** | **109.3** | +26% (unthrottled −47%) |

수렴: price→6.92, MBA 10~20% 진동, **총 BW→target 5000 수렴**(4655~5453).

## 판정
1. **MERCATO 작동**: 가격 제어로 BW 를 target 으로 끌고 victim 보호 (CSMA 238 의 109.7
   과 동급). actuator = MBA(연속 BW 레버, 유효).
2. **연속 proportional vs CSMA binary**: MBA 10% grain 으로 부드러운 제어, 가격이 혼잡
   반영. 다중 harvest worker 가 같은 가격에 반응 → proportional-fair 분배 (시장 균형).
3. CC-CAT(기각)과 대조 — actuator 가 BW(MBA)라 유효.

## 비고
- victim 완전보호 아님(+26%): target>victim_BW 헤드룸 + MBA 의 연성 throttle. target↓로 강화.
- 다중-worker 공정성(Jain)은 단일 worker 데모라 미측정 (시장 메커니즘상 성립 예상).

산출물: `mercato_ctl.py`, `runs/`.
