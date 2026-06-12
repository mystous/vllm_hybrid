# SUB_246 — [A10] IAA-SQUEEZE: 압축 대역 굴절

> **상태**: 대기 — 2차 알고리즘·압축 ⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 디바이스/정보이론
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

harvest 입력을 압축 상태로 저장·운반, IAA (실측 로드) 가 소비 직전 해제 — 공유 버스 바이트 1/r, 코어 사이클 0. 압축을 *간섭 절감* 목적으로 사용.

## 가설 / 메커니즘

DRAM/UPI 트래픽 = 압축 크기. r=3 텍스트류면 mbm_total 1/3. r 낮은 데이터는 IE 게이트로 작업별 채택. FERRY 합성 시 UPI 레그도 1/r.

## 실험 설계

T1 — 비압축 스트리밍 vs SQUEEZE (r≥2 데이터), mbm_total + victim p99 + 유용 처리량.

## 게이트

동일 처리량에서 mbm_total ≥−30% AND victim p99 ≥+2% 개선.

## 의존 / 비고

선행: IAA WQ 구성 (SUB_236 류), SUB_227 (IE). GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A10`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
