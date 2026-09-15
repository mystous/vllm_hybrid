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
- [x] RESULT.md §B.2~B.6

## TSK_051 — 실험 A: 최대 모델 Kimi-K2-Instruct

- [x] 다운로드 1,030 GB → `~/.cache/huggingface` (2 h 45 min)
- [x] `kt quant -m int4 -i fp8` — 68.6 min, 488 GB
- [x] 컨테이너 추가 불필요 (같은 캐시 경로)
- [x] 하이브리드 TP8 부팅 240 s → greedy 4/4 → C8 22.78 tok/s, DRAM 576 GB, HBM 13 GiB/장
- [x] 품질 통과 — SUB_167 결함 재현 안 됨 (가설 축소 기록)
- [x] RESULT.md §A.4 용량 한계 표
