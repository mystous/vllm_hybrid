# SUB_224 — [D11] SW 자가계측 token-bucket (소프트웨어 MBA)

> **상태**: 대기 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 메모리 BW
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

epoch(1ms)당 자기 바이트 카운트 → budget 초과 시 잔여 epoch tpause — resctrl 없는 VM/AMD/ARM 일반화 메커니즘.

## 가설 / 메커니즘

MBA 와 같은 frontier 에 올릴 수 있는 보편 소프트웨어 대체재가 존재한다.

## 실험 설계

budget {25,50,75}% vs MBA {20,50,80}% 동일 frontier 비교 + 정확도 (목표 vs mbm 실측)·오버헤드.

## 게이트

MBA frontier 와의 격차 ≤10%p → enforcement ladder 가운데 칸 확정.

## 의존 / 비고

victim_aggressor.c +50줄. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D11`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
