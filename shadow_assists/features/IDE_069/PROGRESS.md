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

## 2026-09-15 14:48 — sweep 2라운드 완료
| 셀 | 구성 | C32 tok/s | C64 tok/s | TPOT p50 (C32) | 비고 |
|---|---|---|---|---|---|
| **t7** | hot96 def4, mf 0.95, **KV 40,960** | 459.34 | **637.24** | 52.2 ms | C64 신기록 (24k KV 때 425.15) |
| t8 | + EAGLE3 spec (480B 전용 draft) | 293.46 | — | 95.6 ms | 악화 — 검증 토큰이 CPU expert 비용을 키움 |
| t9 | + STANDALONE spec (Qwen3-4B) | 169.77 | — | 25.4 ms (TTFT 20 s) | 악화 |
| t10 | cpuinfer 112 | 483.24 | — | 49.7 ms | 96 대비 차이 없음/소폭 하락 |
| t11 | dispatch static | 엔진 거부 | — | — | `ValueError` static 은 a2a 백엔드 전제 |
| t12 | hot100, KV 16,384 | 356.45 | — | 65.8 ms | KV 압박(TTFT p95 7.2 s) 이 hot 4개 이득을 상쇄 |
- **최고 = t7 @C64 637.24 tok/s** (출발점 43.39 대비 **14.7×**, IDE_030 C64 408.2 대비 +56 %). greedy 전 셀 4/4.
- spec decode 두 종은 이 구성에서 역효과 — CPU expert 가 지배하는 스텝에 draft 검증 토큰이 더해져 CPU 비용이 커진다.
- 같은 구성의 C32 가 run 마다 454~520 으로 흔들린다 (±7 %) → 3라운드에서 반복 측정으로 jitter 를 잡고 C48/80/96 을 본다. dual 비교 (TSK_053) 가 먼저 실행 중.

## 2026-09-15 15:10 — dual 1차: 기준선 확보, 컨테이너 복제 실패 → 재사용으로 전환
- **GPU-only TP8+EP8 기준선 (`20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/`)**: C32 **1,029.75 tok/s** (15.9 s), C64 **2,031.11 tok/s** (16.1 s). greedy 4/4. GPU-only 는 C 에 비례해 커진다 (C16 701.6 → C32 1,030 → C64 2,031).
- `docker commit sgl-kt` 가 nerdctl snapshot mount 오류로 실패 (`failed to export layer: mount callback failed`, 14 분 소요 후) → 컨테이너 미생성, dual 두 셀 rc=1.
- 전환: 8-30 dual 실험의 cpuset 컨테이너 재사용 — **sgl-kt5 = 소켓0 (0-55,112-167 / mems 0), sgl-kt2 = 소켓1 (56-111,168-223 / mems 1)**, kt-kernel 0.7.0.post2 + 패치 2건 확인. sgl-kt(post1 소스빌드)와 소프트웨어가 다르므로 dual 앞에 **sgl-kt5 단독 hot96 def4 C32** 를 먼저 재어 t2(519.8)와의 등가성을 확인한다.
- 3라운드(jitter·C sweep) 종료 후 dual v2 자동.

## 2026-09-15 15:24 — 라우터 단일 엔드포인트 셀 추가 (사용자 지시)
- 사용자: "모델 인스턴스 2개가 뜨면 따로따로 동작시켜서 출력을 합쳐서 보여 줄 거냐" → 합산 방식(독립 포트 2개, 동시 벤치, tok/s 합산) 설명 후 라우터 실험도 진행 지시.
- `run_router.sh`: sglang_router 0.3.2 (`--worker-urls :30000 :30001 --policy round_robin|cache_aware`) 를 sgl-kt5 에서 :30002 로 띄우고 단일 엔드포인트로 C32/C64 벤치. dual v2 종료 후 자동.

## 2026-09-15 15:30 — sweep 3라운드 완료 (반복·C sweep)
| 구성 | C32 | C48 | C64 | C80/96 |
|---|---|---|---|---|
| hot96 def4 KV40k rep1 (t13) | 465.20 | 564.57 | 637.54 | C80 **239.77** (붕괴, TPOT 297 ms) |
| hot96 def4 KV40k rep2 (t14) | 471.90 | — | **650.75** | — |
| hot96 def4 KV40k (t7, 2라운드) | 459.34 | — | 637.24 | — |
| hot96 def4 KV56k (t15) | — | — | 634.06 | C96 328.72 (붕괴) |
| hot96 def4 KV24k rep (t16) | 486.90 | — | — | — |
| hot96 def4 KV24k (t2, 1라운드) | 519.81 | — | 425.15 | — |
- **jitter**: 같은 구성 반복에서 C32 ±1.5 % (459/465/472), C64 ±1.1 % (637/638/651). 1라운드의 454~520 폭은 대부분 KV 설정 차이(24k 대 40k)였다.
- **최고 = hot96 def4 KV40k @C64: 642 ± 7 tok/s (3회)**. 출발점 43.39 대비 **14.8×**.
- KV pool 이 C 를 정한다: 24k 는 C32 까지(C64 425), 40k 는 C64 까지(C80 붕괴), 56k 도 C96 은 붕괴. KV 를 늘리면 C32 처리량은 소폭 떨어진다 (520/487 → 465, mem-fraction 0.92→0.95 로 graph/workspace 여유가 줄어드는 것으로 추정, 미검증).
- dual v2 진행 중 (sgl-kt5 단독 검증 셀 부팅).

## 2026-09-15 15:53 — dual v2 완료 (cpuset 컨테이너 sgl-kt5/sgl-kt2, kt-kernel 0.7.0.post2)
| 구성 | C16each 합산 | C32each 합산 | 인스턴스별 |
|---|---|---|---|
| dual expert 0 (cpuinfer 48×2) | **49.19** | **66.01** | 24.8+24.4 / 32.7+33.3 |
| dual hot96 def4 (cpuinfer 48×2) | **512.48** | **691.25** | 260.6+251.9 / 347.9+343.4 |
| GPU-only TP8+EP8 (같은 총부하) | 1,029.75 (C32) | 2,031.11 (C64) | — |
- 두 인스턴스는 완전 대칭 (±2 %), greedy 양쪽 4/4, 간섭 없음. expert 0 dual 은 단일 4장(43 @C16 / 8-30 66.6 @C32)과 같다 — CPU 대역폭을 둘로 나눈 만큼 각자 절반.
- **⚠ 컨테이너 등가성 미달**: sgl-kt5 단독 hot96 def4 cpuinfer 96 C32 = **336.27** vs sgl-kt 의 같은 구성 519.8/486.9 → sgl-kt5/sgl-kt2 (kt-kernel 0.7.0.post2 정식판) 가 sgl-kt (post1 소스빌드 + sglang 로컬 수정 3파일 = IDE_033 계열 개선) 보다 **약 30 % 느리다**. 따라서 위 dual 수치는 느린 소프트웨어 기준이며, TP8 과의 격차 일부는 소프트웨어 차이다. 라우터 셀 종료 후 sgl-kt 의 kt_kernel·sglang 수정본을 두 컨테이너에 동기화하고 dual hot 을 재측정한다.
