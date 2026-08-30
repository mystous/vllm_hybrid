# spec×graph 결합 공사 노트 (2026-08-30 저녁)

## 목표
초안검증(4B draft, K=3)과 decode CUDA graph 의 동시 사용. 단독 기여: spec +20%(pre-graph 실측) / graph +48%(spec 대비). 결합 시 기대 이득: +10% 이하로 추정 (graph 후 TPOT 82ms 체제에서 draft 상대 비용 증가).

## 확정된 사실 (로그·코드 판독)
1. spec 모드에서 target verify graph 는 **캡처는 정상** ("target verify, num_tokens_per_req=4", 21s, 실패 없음 — /tmp/sgl_pw.log)
2. 죽는 곳은 **재생(replay)**: `full_cuda_graph_backend.replay → CUDA error: invalid argument` (부하 시)
3. 일반 decode graph (spec 없음, 동일 kt 구성) 는 캡처·재생 모두 정상 (+475% 의 기반) → **kt 는 무죄**
4. 남는 용의자: draft decode graph 와 target verify graph 의 **연쇄 재생** 시 스트림/버퍼 상태 불일치 (draft graph 재생 직후 target graph 재생이 이어지는 유일한 차이점)

## 다음 작업 목록 (미착수)
- [ ] CUDA_LOG_FILE=stderr 로 재현해 invalid argument 의 대상 API 특정
- [ ] draft 쪽 graph 만 비활성 (코드: eagle_worker_v2.init_cuda_graphs(capture_decode_cuda_graph=False) 경로 강제) 후 target verify graph 단독 재생 시험 → 연쇄 가설 판별
- [ ] verify replay 직전 forward_batch.spec_info.num_tokens_per_req 채워지는지 (guard 동작) 확인
- [ ] 통과 시: G80 구성 + spec 재벤치, 이득 ≥+5% 게이트

## 참고
- turbo 해제는 OS 에서 불가 (no_turbo 쓰기 root 도 거부 — BIOS 잠금, 하드웨어 관리자 필요) — 보류 기록
