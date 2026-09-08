# PLN_008 M3 준비 — Qwen3-235B-A22B 하이브리드 서빙 스모크 (2026-09-08 20:43)
- 변환: FP8 → kt AMXINT4 (94층, 128 expert, moe_inter 1536, hidden 4096, kv-head 4). 115GB, CONVERT_EXIT=0.
- 부팅: TP=4 H100 + hot-32 + AMXINT4 96스레드, mem-fraction 0.90, max-total-tokens 32768. HEALTH 130s.
- 품질: greedy "The capital of France is" → "Paris, which is also the largest city" (정상).
- 부하: 16 동시 버스트 16/16 성공(8.5s). **하이브리드 서빙 성립.**
- M3 남은 작업: 235B 라우팅 트레이스(expert 빈도) 수집 → hotmap → hot-H 스윕, 30B 반례, 480B 주력으로 3모델×워크로드 표. hot-32 는 임시값(235B 라우팅 트레이스 없어 균등 가정).
