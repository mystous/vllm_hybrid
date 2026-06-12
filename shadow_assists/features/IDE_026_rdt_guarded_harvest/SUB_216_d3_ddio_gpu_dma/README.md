# SUB_216 — [D3] DDIO way × GPU DMA 경합

> **상태**: ❌ 기각 (2026-06-12 범위 재정의) | **parent**: `TSK_048` (`IDE_026`) | **수준**: LLC/IO
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

L3 way 18-19 = IO(DDIO) 공유 실측 (`shareable_bits=c0000`). PCIe inbound 가 이 way 에 착지.

## 가설 / 메커니즘

측정 목적이 *GPU 전송 성능* — IDE_026 범위 (비-GPU 하드웨어 최적화) 밖이므로 기각.

## 실험 설계

(기각 — 실험 없음)

## 게이트

(기각)

## 의존 / 비고

부산물 존치: 'harvest mask 는 way 18-19 비침범' = task.md T1 배치 규칙 #3 (방어적 제약, 측정 불요).

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D3`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
