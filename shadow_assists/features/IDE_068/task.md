# IDE_068 task

## TSK_050 — 실험 B: 480B 오프로딩 GPU 최소 장수

- [x] b0  GPU-only TP8 + EP8 기준선 — 701.60 tok/s, greedy 4/4
- [x] b0a GPU-only TP4 — OOM 40초
- [x] b1  하이브리드 TP4 — 43.39 tok/s
- [x] b2  하이브리드 TP2 — 43.33 tok/s
- [x] b3  하이브리드 TP1 — 43.24 tok/s, HBM 24 GiB
- [x] b4~b6 남는 HBM 에 expert 16/40/96 배치 — +8/+15/+19 % (hotmap 없는 하한)
- [x] b7~b8 TP1 cuda graph ON — 효과 없음 (±1 %)
- [x] b9~b14 GPU 3·5·6·7 장 — 6/6 엔진 거부 (vocab 151936·expert 160 정수 제약). DP 대안(B5)은 사용자 지시로 중단
- [ ] RESULT.md — 장수별 성립·처리량·지연·품질·HBM 사용량 표

## TSK_051 — 실험 A: 최대 모델 Kimi-K2-Instruct

- [ ] 다운로드 1.03 TB → `/data/hf` (백그라운드)
- [ ] `kt quant -m int4 -i fp8` 변환 (GPU 실험과 CPU 경합 — 실험 B 셀 사이 또는 이후)
- [ ] 컨테이너 `sgl-kt6` 생성 (`/data/hf:/models2` 마운트, 패치 재적용)
- [ ] 하이브리드 TP8 부팅 → greedy 4문항 → C8 벤치 + DRAM·HBM 사용량
- [ ] 품질 미통과 시 `SUB_167` 연계 기록
- [ ] RESULT.md — 용량 한계 표 (DRAM 2 TB 대비 사용량, 이론 상한)
