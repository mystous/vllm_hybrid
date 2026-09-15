# IDE_069 task (TSK_052)
- [ ] hotmap 재생성 — recorder per_pass 로 sonnet 라우팅 트레이스 → 층별 빈도 → hotmap.json
- [ ] t1 hot-96 + graph ON (deferral 0) C16/C32
- [ ] t2 hot-96 + graph + deferral 4 (IDE_030 재현) C16/C32/C64
- [ ] t3 hot-112 (mem-fraction 0.95, max-total-tokens 축소)
- [ ] t4 deferral 2 / 8 sweep
- [ ] t5 dispatch dynamic on/off
- [ ] t6 spec decode (EAGLE3 draft 보유 / STANDALONE Qwen3-4B)
- [ ] 품질 게이트: greedy 4/4 전 셀, GSM 40 (deferral 셀)
- [ ] RESULT.md

## TSK_053 — TP4+오프로딩 ×2 대 GPU-only TP8 (사용자 지시)
- [ ] GPU-only TP8+EP8 C32/C64 기준선 (b0 는 C16 만)
- [ ] 소켓1 cpuset 컨테이너 준비 (패치 4건 재적용)
- [ ] dual: A(0-3, 소켓0) + B(4-7, 소켓1), hot96 def4 graph, cpuinfer 48 — 동시 C16/C32 each
- [ ] 합산 vs TP8 비교표
