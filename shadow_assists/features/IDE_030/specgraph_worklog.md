# spec×graph 결합 공사 노트 (2026-08-30 저녁)

## 목표
초안검증(4B draft, K=3)과 decode CUDA graph 의 동시 사용. 단독 기여: spec +20%(pre-graph 실측) / graph +48%(spec 대비). 결합 시 기대 이득: +10% 이하로 추정 (graph 후 TPOT 82ms 체제에서 draft 상대 비용 증가).

## 확정된 사실 (로그·코드 판독)
1. spec 모드에서 target verify graph 는 **캡처는 정상** ("target verify, num_tokens_per_req=4", 21s, 실패 없음 — /tmp/sgl_pw.log)
2. 죽는 곳은 **재생(replay)**: `full_cuda_graph_backend.replay → CUDA error: invalid argument` (부하 시)
3. 일반 decode graph (spec 없음, 동일 kt 구성) 는 캡처·재생 모두 정상 (+475% 의 기반) → **kt 는 무죄**
4. 남는 용의자: draft decode graph 와 target verify graph 의 **연쇄 재생** 시 스트림/버퍼 상태 불일치 (draft graph 재생 직후 target graph 재생이 이어지는 유일한 차이점)

## 진행 (저녁 2차)
- [x] draft graph 비활성 스위치 (SGL_SKIP_DRAFT_GRAPH env, eagle_worker_v2 패치) → draft 캡처 0건 확인
- [x] **연쇄 가설 기각**: draft eager 상태에서도 부하 시 동일 segfault → 범인 = 폭4 verify graph 재생 자체 × kt 상호작용 (폭1 decode graph 는 kt 와 정상 공존 — 폭이 유일한 차이)
- 스모크(단건)는 정상, 동시 부하에서만 segfault → 배치 크기 의존

## 다음 작업 목록
- [ ] CUDA_LOG_FILE=stderr + 코어 트레이스로 segfault 지점 특정 (kt submit 버퍼 폭? chunked prefill 경계?)
- [ ] kt wrapper 의 verify 경로 버퍼 크기 (T=4×C) 가 capture 시 고정 버퍼와 어긋나는지 코드 확인
- [ ] 판정 유지: 성공 시 이득 ≥+5% 게이트, 실패 누적 시 공사 중단 보고 (기대이득 ≤+10% 대비 비용 재평가)

## 참고
- turbo 해제는 OS 에서 불가 (no_turbo 쓰기 root 도 거부 — BIOS 잠금, 하드웨어 관리자 필요) — 보류 기록
