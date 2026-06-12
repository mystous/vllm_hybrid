# SUB_236 — [D23] DSA/IAA 트래픽 셰이핑 (read buffer·WQ·token-bucket)

> **상태**: 대기 — 채널② 실물 제어 ⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 디바이스 (비-GPU 가속기)
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

실측 노브: group read_buffers_allowed=96(max)/use_read_buffer_limit=0/traffic_class_a,b + WQ priority/size. read buffer = DSA in-flight 메모리 동시성의 직접 상한.

## 가설 / 메커니즘

read buffer 96→24 면 DSA 발 트래픽 (채널 ②, MBA 사각) 피크가 ~¼ 상한 — RDT 가 못 막는 간섭을 디바이스 측에서 막는다.

## 실험 설계

harvest WQ 별도 group 분리 + read_buffers {96,48,24,12} sweep × DSA memcpy aggressor vs victim → 간섭 상한 곡선. 동적: ENQCMD token-bucket.

## 게이트

N=24 에서 victim p99 회복 ≥70% AND DSA 처리량 ≥ 무제한의 50%.

## 의존 / 비고

IDE_023 dsa_lane 재사용. IAA 에도 동일 적용. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3c D23`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
