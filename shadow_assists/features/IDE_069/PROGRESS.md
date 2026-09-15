# IDE_069 진행 로그

## 2026-09-15 13:49 — 착수
- IDE_069/TSK_052/TST_026 발급, 브랜치 feat/480b-tp4-hot-expert (feat/moe-offload-limits 에서 분기).
- 출발점 IDE_068 b1 (TP4, 43.39 tok/s). 참조 IDE_030 hot-96+def4 C32 490.9.

## 2026-09-15 14:00 — hotmap 재생성 완료, sweep 1라운드 착수
- recorder `stat` 모드, TP4 하이브리드(expert 0) 에 sonnet 512/128 C16 64req 흘림 → `logical_count (1000 step × 62층 × 160)`, 총 **71,981,504** expert 호출. `eval/results/20260915_135132_ide069_hotmap/`.
- hotmap 커버리지 (층 평균 / 최소): top-64 94.1 / 77.1 %, **top-80 97.0 / 85.4 %** (IDE_030 기록 97.0 % 과 일치), **top-96 98.6 / 91.4 %**, top-112 99.5 / 95.4 %, top-128 99.9 / 98.0 %.
- hotmap.json = `~/.cache/huggingface/kt/ide069/hotmap.json` (컨테이너 `/models/kt/ide069/hotmap.json`). 사본을 결과 디렉터리에 보관.
- sweep 1라운드 (`cells_round1.txt`, 6셀): t1 hot96 def0 graph / t2 hot96 def4 (IDE_030 재현) / t3 hot112 def4 / t4 def2 / t5 def8 / t6 hot64 def4. 모두 dispatch dynamic, C16/32(/64).

## 2026-09-15 14:04 — ★ t1 hot96 (hotmap) + cuda graph, deferral 0
- 부팅 HEALTH OK 110 s. HBM **74.6 GiB/장** (17 → 75). greedy 정상.
- **C16 321.98 tok/s** (TTFT p50 1,224 ms, TPOT p50 39.5 ms) · **C32 470.26 tok/s** (TTFT 1,394 ms, TPOT 55.3 ms). CPU busy 평균 32~34 %.
- 출발점(IDE_068 b1, 43.39 @C16) 대비 **7.4×** (C16). IDE_068 b6 의 물리 id 96개(51.55)와의 차이 = hotmap 의 효과 (6.2×).
- deferral 없이 이미 IDE_030 의 def4 기록(490.9 @C32)의 96 % 수준. t2 (def4) 진행 중.

## 2026-09-15 14:09 — ★ t2 hot96 + deferral 4 + graph (IDE_030 처방 재현·초과)
- 부팅 100 s, HBM 74.6 GiB/장, greedy 4/4.
- **C16 329.13** (TPOT 38.5 ms) · **C32 519.81 tok/s** (TTFT p50 1,396 ms, TPOT p50 49.2 ms) · C64 425.15 (TTFT p95 15.5 s — KV pool 24,576 토큰 압박).
- IDE_030 기록 (C32 490.9 / C64 408.2) 을 **재현하고 +5.9 % 초과**. 출발점 43.39 (C16) 대비 C32 기준 **12.0×**.
- deferral 4 의 기여: C32 470.26 → 519.81 (+10.5 %), C16 +2.2 %. 근사 연산이므로 GSM 40 게이트 예정.
- C64 는 KV 상한(max-total-tokens 24,576 < 64×640) 에 걸림 → 2라운드에서 KV 확대 대 hot expert 수의 trade-off 를 본다.

## 2026-09-15 14:22 — sweep 1라운드 완료
| 셀 | 구성 | C32 tok/s | TPOT p50 | HBM/장 |
|---|---|---|---|---|
| t1 | hot96 def0 graph | 470.26 | 55.3 ms | 74.6 GiB |
| **t2** | **hot96 def4 graph** | **519.81** | 49.2 ms | 74.6 GiB |
| t3 | hot112 def4 mf0.95 | **OOM 40 s** | — | — |
| t4 | hot96 def2 | 454.27 | 53.8 ms | 74.6 GiB |
| t5 | hot96 def8 | 500.17 | 47.3 ms | 74.6 GiB |
| t6 | hot64 def4 | 230.14 | 119.3 ms | 53.1 GiB |
- 최고 = t2 (hot96, deferral 4): **519.81 tok/s @C32**. deferral 은 4 가 최적 (0/2/4/8 = 470/454/520/500).
- hot 수가 지배 항: 64 → 96 에서 2.26× (230 → 520). hot112 는 산술상 불가 — TP4 샤드 기준 expert 1개/층 ≈ 11.8 MB × 62층 = 0.73 GB/expert → 112개 = 82 GB > 79.2 GB 용량 (KV 를 0 으로 해도 안 들어감). t3 OOM 이 이를 확인.
- C64 는 KV 상한 → 2라운드 t7 (mf 0.95, max-total-tokens 40,960).
- 2라운드 (GSM 게이트 뒤 자동): t7 KV 확대 / t8 EAGLE3 spec / t9 STANDALONE(Qwen3-4B) spec / t10 cpuinfer 112 / t11 dispatch static / t12 hot100 (상한 탐침).

## 2026-09-15 14:33 — GSM 40 게이트 통과, dual 비교 준비
- **GSM8K 40문항 (hot96, chat greedy)**: deferral 0 = **39/40 (97.5 %)**, deferral 4 = **39/40 (97.5 %)** → 저하 0. t2 구성(def4) 채택 가능. `eval/results/20260915_141917_ide069_gsm40_hot96/`.
- 사용자 추가 지시: TP4+오프로딩 ×2 (GPU 8장) 대 GPU-only TP8 비교, 오프로딩은 **hot expert 상주(hot96+def4)·비상주(expert 0) 둘 다**. `run_dual.sh` 작성 — 인스턴스별 cgroup cpuset 컨테이너(소켓0/1, sgl-kt 를 이미지로 커밋해 동일 상태), cpuinfer 48·threadpool 2 (8-30 교정판 처방), 동시 벤치 C16each/C32each 합산 vs TP8 C32/C64. 2라운드 종료 후 자동.
