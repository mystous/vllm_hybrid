# SUB_239 — [A3] FERRY: DSA-운반 NUMA 파이프라인

> **상태**: 대기 — 신규 알고리즘 ⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: NUMA/디바이스
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

'컴퓨트는 로컬, 운반은 디바이스' — UPI 레그를 DSA (read_buffers=24 셰이핑) 가 전담, 코어/AMX 는 2MB-THP 바운스 버퍼만 접근. 채널 ④(MBA 사각)→① 변환기.

## 가설 / 메커니즘

더블버퍼 F(i)∥C(i−1), 완료 대기 umonitor/umwait (폴링 트래픽 0). 밸런스: T_copy/N_rb ≤ T_compute.

## 실험 설계

동일 원격 데이터 처리량 기준 — 코어 직접 원격 vs FERRY, victim p99 + 처리량 + N_rb sweep frontier.

## 게이트

직접 원격 대비 p99 영향 ≤½ AND 처리량 ≥80% AND frontier 단조성.

## 의존 / 비고

선행: SUB_236 (read buffer 곡선), SUB_234 (THP). GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A3`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
