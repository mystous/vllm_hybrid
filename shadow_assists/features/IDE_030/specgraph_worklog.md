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

## 종결 (2026-08-30 밤)

1. -1 id 가설 기각 (프로브 실측: 음수 id 0건). 크래시는 부하 무관, **verify graph 첫 재생에서 즉사**로 특정
2. **근본 원인 (버그 #5)**: sglang 이 kt 에 캡처 배치 목록을 요청-폭 1 기준으로 전달 (`set_capture_batch_sizes(self.capture_bs)`). 검증 graph 는 요청당 4토큰이라 실제 크기 = bs×4 인데 목록에 없어, kt 가 임시 버퍼를 쓰고 다음 캡처 때 재할당 → graph 가 해제된 고정 버퍼를 재생에서 접근 → segfault
3. 수리 (3줄): 목록에 `captured_req_width` 곱을 반영 → **spec+graph 전체 부하 생존, 품질 정상**
4. **성능 판정: 결합 기각** — C=32: 279.8 (graph-only 324, −13.6%) / C=16: 206.4 (251.5, −18%). accept 0.67 로 좋아도 graph 체제 (TPOT 82ms) 에선 draft 왕복 비용이 검증 절약을 상회
5. 산출물: 성능 0 / 안정성 수리 1건 (upstream 제보 #5 — spec+graph+kt 사용자의 확정 크래시)

**최종 확정 구성 유지: hot-80 + full decode graph, spec 없음 = 324 tok/s @C32**
