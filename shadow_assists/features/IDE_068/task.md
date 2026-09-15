# IDE_068 task

## TSK_050 — 실험 B: 480B 오프로딩 GPU 최소 장수

- [ ] b0  GPU-only TP8 + EP8 기준선 — 부팅, greedy 4문항, C16 벤치
- [ ] b0a GPU-only TP4 — OOM 재실증 (TSK_047 재현, 45초 예상)
- [ ] b1  하이브리드 TP4, expert 전량 CPU — 부팅, greedy, C16 벤치 (TSK_047 44.5 tok/s 대조)
- [ ] b2  하이브리드 TP2
- [ ] b3  하이브리드 TP1
- [ ] (선택) 성립한 최소 TP 에서 남는 HBM 에 hot expert 배치 — 처리량 회복 여부
- [ ] RESULT.md — 장수별 성립·처리량·지연·품질·HBM 사용량 표

## TSK_051 — 실험 A: 최대 모델 Kimi-K2-Instruct

- [ ] 다운로드 1.03 TB → `/data/hf` (백그라운드)
- [ ] `kt quant -m int4 -i fp8` 변환 (GPU 실험과 CPU 경합 — 실험 B 셀 사이 또는 이후)
- [ ] 컨테이너 `sgl-kt6` 생성 (`/data/hf:/models2` 마운트, 패치 재적용)
- [ ] 하이브리드 TP8 부팅 → greedy 4문항 → C8 벤치 + DRAM·HBM 사용량
- [ ] 품질 미통과 시 `SUB_167` 연계 기록
- [ ] RESULT.md — 용량 한계 표 (DRAM 2 TB 대비 사용량, 이론 상한)
