# SUB_235 — [D22] NUMA 배치·드리프트 복구 알고리즘

> **상태**: 대기 — NUMA 매크로 | **parent**: `TSK_048` (`IDE_026`) | **수준**: NUMA/UPI
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

first-touch 탓에 harvest 메모리가 remote 에 잡히면 UPI 경유 (MBA 사각). 정적 membind 는 장기 드리프트를 못 막음.

## 가설 / 메커니즘

remote_ratio(>0.2, 3 epoch) → ① MPOL_BIND 재고정 ② move_pages() hot 버퍼 이주 (비용모델 가드) ③ 잔존 시 governor DUTY 강제 — 의 3단 복구 루프.

## 실험 설계

의도적 remote 배치 → 복구 ON/OFF — rr 시계열 수렴 속도 + victim p99.

## 게이트

수렴 ≤ 1s AND 이주 비용 < 절감 트래픽 (비용모델 검증).

## 의존 / 비고

T1+libnuma. 정적 N8/SUB_165 지식의 동적 승격. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3c D22`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
