# IDE_069 진행 로그

## 2026-09-15 13:49 — 착수
- IDE_069/TSK_052/TST_026 발급, 브랜치 feat/480b-tp4-hot-expert (feat/moe-offload-limits 에서 분기).
- 출발점 IDE_068 b1 (TP4, 43.39 tok/s). 참조 IDE_030 hot-96+def4 C32 490.9.

## 2026-09-15 14:00 — hotmap 재생성 완료, sweep 1라운드 착수
- recorder `stat` 모드, TP4 하이브리드(expert 0) 에 sonnet 512/128 C16 64req 흘림 → `logical_count (1000 step × 62층 × 160)`, 총 **71,981,504** expert 호출. `eval/results/20260915_135132_ide069_hotmap/`.
- hotmap 커버리지 (층 평균 / 최소): top-64 94.1 / 77.1 %, **top-80 97.0 / 85.4 %** (IDE_030 기록 97.0 % 과 일치), **top-96 98.6 / 91.4 %**, top-112 99.5 / 95.4 %, top-128 99.9 / 98.0 %.
- hotmap.json = `~/.cache/huggingface/kt/ide069/hotmap.json` (컨테이너 `/models/kt/ide069/hotmap.json`). 사본을 결과 디렉터리에 보관.
- sweep 1라운드 (`cells_round1.txt`, 6셀): t1 hot96 def0 graph / t2 hot96 def4 (IDE_030 재현) / t3 hot112 def4 / t4 def2 / t5 def8 / t6 hot64 def4. 모두 dispatch dynamic, C16/32(/64).
