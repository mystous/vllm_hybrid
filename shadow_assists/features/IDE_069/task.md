# IDE_069 task (TSK_052)
- [x] hotmap 재생성 — recorder per_pass 로 sonnet 라우팅 트레이스 → 층별 빈도 → hotmap.json
- [ ] t1 hot-96 + graph ON (deferral 0) C16/C32
- [ ] t2 hot-96 + graph + deferral 4 (IDE_030 재현) C16/C32/C64
- [ ] t3 hot-112 (mem-fraction 0.95, max-total-tokens 축소)
- [x] t4 deferral 2 / 8 sweep
- [x] t5 dispatch (static 거부) dynamic on/off
- [x] t6 spec (역효과) decode (EAGLE3 draft 보유 / STANDALONE Qwen3-4B)
- [x] 품질 게이트 (GSM40 39/40 = 39/40): greedy 4/4 전 셀, GSM 40 (deferral 셀)
- [x] RESULT.md

## TSK_053 — TP4+오프로딩 ×2 대 GPU-only TP8 (사용자 지시)
- [x] GPU-only TP8+EP8 C32 1029.75 / C64 2031.11 (b0 는 C16 만)
- [x] cpuset 컨테이너 sgl-kt5/2 재사용 + 소프트웨어 동기화 (패치 4건 재적용)
- [x] dual: A(0-3, 소켓0) + B(4-7, 소켓1), hot96 def4 graph, cpuinfer 48 — 동시 C16/C32 each
- [x] 합산 vs TP8 비교표 (RESULT §4)
- [x] **라우터 단일 엔드포인트** (사용자 지시): sglang_router 0.3.2 로 A+B 를 :30002 로 묶어 C32/C64 — expert 0 (round_robin), hot96 def4 (round_robin, cache_aware). 합산 방식과 대조 = 라우터 오버헤드·부하 불균형
