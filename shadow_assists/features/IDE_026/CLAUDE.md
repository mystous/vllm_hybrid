# IDE_026 — Claude 작업 메모

- **실험 실행 금지 상태** (타 실험과 자원 공유) — E1 이상은 사용자 허가 후. E0 (시뮬레이션) 만 예외.
- 스택 기반: 8-27 캠페인의 SGLang+kt-kernel 구축분 (패치 4건 — COMPREHENSIVE_REPORT §3.2 재적용 필요, 컨테이너 휘발).
- 모델: E1~E4 는 Qwen3-30B-A3B (품질 정상 확인). R1 은 SUB_167 해결 전 e2e 사용 금지.
- 착수 시 필수 재조사: CoX-MoE v2 (2605.17889v2) + 2026-06 이후 spec×MoE 신간 — 선점 여부.
- 5월 수락률 실측 (sonnet 0.388 / chat 0.812 / code 0.014, K=7 ngram) 은 α 모델 초기값으로 재사용 (`vllm_config_perf/docs/idea/IDE_011_*`).
