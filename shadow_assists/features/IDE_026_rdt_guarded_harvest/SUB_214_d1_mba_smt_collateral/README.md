# SUB_214 — [D1] `thread_throttle_mode=max` SMT 연좌 throttle 정량화

> **상태**: ✅ 완료 (2026-06-12) | **parent**: `TSK_048` (`IDE_026`) | **수준**: 코어/SMT
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

MBA 는 코어 단위 적용 (실측 `info/MB/thread_throttle_mode=max`) — SMT sibling 이 다른 CLOS 면 가장 강한 throttle 이 물리 코어 전체에 걸린다.

## 가설 / 메커니즘

serving 과 MBA 20% harvest 가 같은 물리 코어의 HT 짝이면 serving 이 20% 로 연좌 throttle → p99 폭증.

## 실험 설계

victim cpu0 고정 + aggressor {cpu16(타 코어), cpu112(sibling)} × MBA {100,20}% = 4셀. 연좌 배율 곡선.

## 게이트

연좌 악화 ≥ +20% → 'vLLM 스레드 배치는 코어-배타 필수' 설계규칙 승격 (논문 §9 design rule).

## 의존 / 비고

T1 인프라 재사용 (+30분). GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D1`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)

## ✅ 결과 (2026-06-12 — v2, 70B × 7 corpus × 4셀, 셀별 fresh boot)

| 조합 | serving (C0 대비, 기하평균) | harvest |
|---|---:|---:|
| C0 간섭 없음 | 100% | 0 |
| **C3 sibling+MBA20** | **99.0%** | 16.2 GB/s |
| C1 분리코어+MBA20 | 97.9% | 16.5 GB/s |
| C2 sibling 무제한 | 83.9% | 75.6 GB/s |

1. **연좌 throttle 미발생** — C3/C2 = +18%, C3 ≈ C0. D1 게이트 (연좌 ≥+20%) 불충족
   → **코어-배타 강제 불필요**. 해석: vLLM 호스트 스레드 96-100% sleep → 노출 미미.
2. **신규 규칙**: "sibling harvest 는 MBA 가드 필수" (무가드 −16.1% → 가드 −1.0%).
3. **binding 간섭 = 메모리 BW** (CAT 불요, LLC 288MB 점유에도 무해).
4. MBA 캘리브레이션 1점: 설정 20% → 실효 21.4% (75.6→16.2 GB/s).
5. **방법론 발견**: 단일 부팅 셀 비교는 suffix global tree 누적학습으로 최대 +24%
   오염 (acc 0.72→0.86) → 전 SUB 에 "셀별 fresh boot" 규칙. v1 증거 보존.

상세: `MEASUREMENTS.md` / 데이터: `runs/` (v2 본판정), `runs_v1_singleboot/` (드리프트 증거)
