# SUB_225 — [D12] Python 런타임 간섭 채널 ⑥ 분리 (GIL·allocator·프로세스 경계)

> **상태**: 대기 — ⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: OS/런타임
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

C-8a(−0.35%) 의 원인 후보 = 메모리 계층이 아닌 Python 런타임 공유 상태 (GIL/allocator/runqueue) — RDT 가 원천적으로 못 보는 채널.

## 가설 / 메커니즘

in-process Python 스레드 harvest 는 GIL hold 만큼 victim 이벤트루프를 직접 지연 — RDT ON 에도 잔존. 프로세스 분리가 유일 해법이면 설계규칙 승격.

## 실험 설계

{C-스레드, Python-스레드, 별도 프로세스} 3단 × RDT {ON,OFF} × allocator {glibc, mimalloc LD_PRELOAD}. victim = asyncio 이벤트루프 모사.

## 게이트

'RDT 가 못 줄이는 잔여' = 채널 ⑥ 크기 정량. 명확 시 'harvest 는 프로세스 경계 밖' 규칙.

## 의존 / 비고

합성판 GPU 불요 / vLLM 판은 T2 무대. free-threaded 3.13t 비교는 dev 머신.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D12`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
