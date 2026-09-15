# IDE_023 / TSK_043 — 원시 데이터 전문

CPU MoE Expert Offloading 백서 (`IDE_023_TSK_043_CPU_MoE_Offloading_백서.docx`) 가 근거로 삼은 원시 데이터 전부다. 발췌하지 않았다.

## 1. 문서 정보

| 항목 | 값 |
|---|---|
| 대상 트랙 | IDE_023 / TSK_043 — MoE Expert Offload |
| 상위 캠페인 | PLN_003 — Hybrid Regime Sweep |
| 측정일 | 2026-08-27 |
| 측정 노드 | violet-h100-016 — Xeon Platinum 8480+ ×2 (AMX) / DDR5 2 TB / H100 80 GB ×8 |
| CPU 클럭 | 2.0 GHz 고정 (turbo OFF, `no_turbo=1`) |
| 커밋 | `437c69ae5` (branch `feat/hybrid-regime-sweep`) |
| 생성 스크립트 | `shadow_assists/features/IDE_023/whitepaper/make_rawdata_md.py` |

### 수록 규칙

- 아래 3개 run 디렉터리의 **모든 파일**을 전문 수록했다. 숨김 파일(`.cpu_mon`, `.gpu_mon` — 샘플러 PID)도 포함한다.
- 파일마다 경로·바이트·줄수·SHA256 을 붙였다. 해시는 원본 바이트 기준이다.
- 파생 집계(§3)는 원문과 분리했고 계산 규칙을 명시했다.
- 원문의 ANSI 이스케이프 시퀀스는 제거했고, 캐리지 리턴(`\r`)은 줄바꿈 + `⏎ ` 로 치환했다. 진행률 표시줄이 한 줄에 겹쳐 기록된 부분을 읽을 수 있게 하기 위한 것이며, 치환 건수는 각 파일 머리에 적었다. 그 외 내용은 그대로다.
- NUL 바이트는 `␀` 로 표시한다.

### run 디렉터리

| 디렉터리 | 내용 |
|---|---|
| `eval/results/20260827_124509_tsk043_smoke_qwen30b/` | Qwen3-30B-A3B smoke — KT 스택 검증 (t1 GPU-only / t2 KT hybrid) |
| `eval/results/20260827_133055_tsk043_main_r1/` | R1-0528 본판 1차 — r0 GPU-only OOM 실증 / r1 FP8 직독 실패 |
| `eval/results/20260827_140008_tsk043_main_r1/` | R1-0528 본판 최종 — r1 AMXINT4 하이브리드 서빙 |

---

## 2. 파일 목록

| # | 경로 | 바이트 | 줄 | SHA256 (앞 16) | 본문 절 |
|---|---|---|---|---|---|
| 1 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/RUN.log` | 425 | 10 | `936e38fdff28490c` | §4 |
| 2 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/.cpu_mon` | 8 | 1 | `af590420629db1b3` | §5 |
| 3 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/.gpu_mon` | 8 | 1 | `b595785797104c7c` | §6 |
| 4 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/bench.log` | 5,309 | 38 | `a29a2fdab2bcbf37` | §7 |
| 5 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/cpu_util.txt` | 399 | 19 | `1eaf41a2c6ef9715` | §8 |
| 6 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/gpu_util.csv` | 1,519 | 64 | `43b943f1432773e7` | §9 |
| 7 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/server.log` | 70,056 | 263 | `03644dbe5399a3e0` | §10 |
| 8 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/smoke.json` | 509 | 1 | `803bd5d641fed295` | §11 |
| 9 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/.cpu_mon` | 8 | 1 | `12d33260a84c5c3f` | §12 |
| 10 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/.gpu_mon` | 8 | 1 | `2e85fbff98393e56` | §13 |
| 11 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/bench.log` | 5,312 | 38 | `80a2254d10f80835` | §14 |
| 12 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/cpu_util.txt` | 3,931 | 179 | `0985cc34398398ab` | §15 |
| 13 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/gpu_util.csv` | 13,798 | 584 | `6a734d67e10d1203` | §16 |
| 14 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/server.log` | 50,564 | 353 | `445e65d3ffa619e7` | §17 |
| 15 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/server_fail.log` | 2,703 | 60 | `b9778e075146e21a` | §18 |
| 16 | `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/smoke.json` | 509 | 1 | `24a3bf847be22d78` | §19 |
| 17 | `eval/results/20260827_133055_tsk043_main_r1/RUN.log` | 279 | 7 | `b292f7f0b035a14e` | §20 |
| 18 | `eval/results/20260827_133055_tsk043_main_r1/r0_gpu_only_oom/oom_evidence.log` | 602 | 5 | `77d2a9f7725ba85c` | §21 |
| 19 | `eval/results/20260827_133055_tsk043_main_r1/r0_gpu_only_oom/server.log` | 112,499 | 737 | `618fad8d459f9c17` | §22 |
| 20 | `eval/results/20260827_133055_tsk043_main_r1/r1_kt_hybrid/server_fail.log` | 18,872 | 120 | `f4297511781379f7` | §23 |
| 21 | `eval/results/20260827_140008_tsk043_main_r1/RESULTS.md` | 2,425 | 40 | `96ef2b9e05d09546` | §24 |
| 22 | `eval/results/20260827_140008_tsk043_main_r1/RUN.log` | 217 | 4 | `79fcbb643ad6a9c6` | §25 |
| 23 | `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/.cpu_mon` | 6 | 1 | `777413895674776b` | §26 |
| 24 | `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/.gpu_mon` | 6 | 1 | `b3845e7e695d8abd` | §27 |
| 25 | `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/bench.log` | 5,062 | 38 | `e26d88ae1e792f30` | §28 |
| 26 | `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/cpu_util.txt` | 2,587 | 118 | `7384b2ee3997011c` | §29 |
| 27 | `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/gpu_util.csv` | 11,306 | 384 | `159b137b0d8d62e0` | §30 |
| 28 | `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/server.log` | 83,346 | 556 | `71819cd9a350bd09` | §31 |
| 29 | `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/smoke.json` | 439 | 1 | `0e80ea9a2fdaea73` | §32 |

합계 **29개 파일 · 392,712 바이트 · 3,626 줄**. 전부 이 문서에 전문 수록되어 있다.

---

## 3. 파생 집계

이 절의 값은 원시 파일에서 **계산한** 것이다. 원문 자체는 §4 이후에 전문으로 있다. 계산 규칙을 각 표 아래에 적었다.

### 3.1 벤치 보고 지표 (bench.log 파싱)

| 지표 | smoke t1 — Qwen3-30B GPU-only | smoke t2 — Qwen3-30B KT hybrid | r1 — R1-0528 KT hybrid (AMXINT4) |
|---|---|---|---|
| 완료 요청 (`Successful requests`) | 128 | 128 | 32 |
| 실패 요청 (`Failed requests`) | 0 | 0 | 0 |
| 최대 동시 요청 (`Maximum request concurrency`) | 16 | 16 | 8 |
| 벤치 소요 s (`Benchmark duration (s)`) | 22.42 | 341.03 | 208.26 |
| 입력 토큰 (`Total input tokens`) | 129773 | 129773 | 32510 |
| 생성 토큰 (`Total generated tokens`) | 32768 | 32768 | 4096 |
| 요청 처리량 req/s (`Request throughput (req/s)`) | 5.71 | 0.38 | 0.15 |
| 출력 처리량 tok/s (`Output token throughput (tok/s)`) | 1461.77 | 96.09 | 19.67 |
| 순간 최대 출력 tok/s (`Peak output token throughput (tok/s)`) | 1719.00 | 128.00 | 32.00 |
| 순간 최대 동시 요청 (`Peak concurrent requests`) | 32.00 | 32.00 | 16.00 |
| 전체 처리량 tok/s (`Total token throughput (tok/s)`) | 7250.88 | 476.62 | 175.77 |
| TTFT 평균 ms (`Mean TTFT (ms)`) | 342.89 | 2596.16 | 12380.89 |
| TTFT 중앙 ms (`Median TTFT (ms)`) | 179.28 | 2381.02 | 11199.23 |
| TTFT p50 ms (`P50 TTFT (ms)`) | 179.28 | 2381.02 | 11199.23 |
| TTFT p95 ms (`P95 TTFT (ms)`) | 1533.03 | 3719.52 | 18098.75 |
| TPOT 평균 ms (`Mean TPOT (ms)`) | 9.64 | 156.98 | 312.46 |
| TPOT 중앙 ms (`Median TPOT (ms)`) | 9.66 | 157.69 | 305.50 |
| TPOT p50 ms (`P50 TPOT (ms)`) | 9.66 | 157.69 | 305.50 |
| TPOT p95 ms (`P95 TPOT (ms)`) | 10.06 | 162.29 | 326.34 |
| ITL 평균 ms (`Mean ITL (ms)`) | 9.64 | 156.98 | 312.46 |
| ITL 중앙 ms (`Median ITL (ms)`) | 9.53 | 154.83 | 304.82 |
| ITL p50 ms (`P50 ITL (ms)`) | 9.53 | 154.83 | 304.82 |
| ITL p95 ms (`P95 ITL (ms)`) | 9.94 | 177.77 | 318.48 |

규칙: `bench.log` 의 `Serving Benchmark Result` 블록에서 지표명으로 시작하는 줄을 찾아 값 부분을 그대로 옮겼다. 반올림·단위 변환을 하지 않았다. `r0` 는 서버 기동 단계에서 실패해 bench.log 가 없다.

### 3.2 CPU 사용률 (cpu_util.txt)

| 셀 | 표본 | 관측 구간 s | 평균 % | 중앙 % | p90 % | 최소 % | 최대 % |
|---|---|---|---|---|---|---|---|
| smoke t1 — Qwen3-30B GPU-only | 19 | 36 | 1.98 | 1.73 | 3.77 | 1.33 | 5.71 |
| smoke t2 — Qwen3-30B KT hybrid | 179 | 357 | 43.06 | 44.77 | 45.31 | 1.82 | 47.74 |
| r1 — R1-0528 KT hybrid (AMXINT4) | 118 | 235 | 42.75 | 46.91 | 47.56 | 3.23 | 49.49 |

규칙: `<unix초> busy=<백분율>` 형식의 줄만 사용. 관측 구간은 첫 표본과 마지막 표본의 시각 차이다. 워밍업 구간을 제외하지 않은 전 구간 집계다. 분위수는 정렬 후 인덱스 방식(보간 없음).

### 3.3 GPU 사용률 (gpu_util.csv)

**smoke t1 — Qwen3-30B GPU-only** — `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/gpu_util.csv`

| GPU | 표본 | util 평균 % | util 최대 % | HBM 평균 MiB | HBM 최대 MiB | 전력 평균 W |
|---|---|---|---|---|---|---|
| 0 | 8 | 48.75 | 100 | 70342 | 70364 | 265.9 |
| 1 | 8 | 0.00 | 0 | 4 | 4 | 68.7 |
| 2 | 8 | 0.00 | 0 | 4 | 4 | 69.1 |
| 3 | 8 | 0.00 | 0 | 4 | 4 | 69.3 |
| 4 | 8 | 0.00 | 0 | 4 | 4 | 69.7 |
| 5 | 8 | 0.00 | 0 | 4 | 4 | 68.8 |
| 6 | 8 | 0.00 | 0 | 4 | 4 | 70.2 |
| 7 | 8 | 0.00 | 0 | 4 | 4 | 70.8 |
| **전체** | **64** | **6.09** | **100** | — | — | — |

**smoke t2 — Qwen3-30B KT hybrid** — `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/gpu_util.csv`

| GPU | 표본 | util 평균 % | util 최대 % | HBM 평균 MiB | HBM 최대 MiB | 전력 평균 W |
|---|---|---|---|---|---|---|
| 0 | 73 | 4.79 | 10 | 70219 | 70320 | 121.5 |
| 1 | 73 | 0.00 | 0 | 4 | 4 | 68.7 |
| 2 | 73 | 0.00 | 0 | 4 | 4 | 69.1 |
| 3 | 73 | 0.00 | 0 | 4 | 4 | 69.4 |
| 4 | 73 | 0.00 | 0 | 4 | 4 | 69.7 |
| 5 | 73 | 0.00 | 0 | 4 | 4 | 68.8 |
| 6 | 73 | 0.00 | 0 | 4 | 4 | 70.2 |
| 7 | 73 | 0.00 | 0 | 4 | 4 | 70.9 |
| **전체** | **584** | **0.60** | **10** | — | — | — |

**r1 — R1-0528 KT hybrid (AMXINT4)** — `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/gpu_util.csv`

| GPU | 표본 | util 평균 % | util 최대 % | HBM 평균 MiB | HBM 최대 MiB | 전력 평균 W |
|---|---|---|---|---|---|---|
| 0 | 48 | 15.54 | 43 | 74706 | 75124 | 154.3 |
| 1 | 48 | 81.06 | 100 | 74086 | 74342 | 174.9 |
| 2 | 48 | 67.21 | 100 | 74086 | 74342 | 166.3 |
| 3 | 48 | 90.85 | 100 | 74086 | 74342 | 191.3 |
| 4 | 48 | 91.00 | 100 | 74086 | 74342 | 189.9 |
| 5 | 48 | 84.29 | 100 | 74086 | 74342 | 175.7 |
| 6 | 48 | 87.10 | 100 | 74086 | 74342 | 184.4 |
| 7 | 48 | 87.65 | 100 | 73606 | 73862 | 186.7 |
| **전체** | **384** | **75.59** | **100** | — | — | — |

규칙: CSV 각 행은 `<index>, <util> %, <mem> MiB, <power> W` 이며 4개 필드가 모두 파싱되는 행만 사용했다. 타임스탬프 열이 없어 벤치 구간으로 창을 좁힐 수 없다 — 전 구간(서버 기동 이후 샘플러 종료까지) 집계다.

### 3.4 CPU 사용률 전체 시계열 (r1)

| # | unix 초 | 경과 s | busy % |
|---|---|---|---|
| 1 | 1787806979 | 0 | 4.69 |
| 2 | 1787806981 | 2 | 16.93 |
| 3 | 1787806983 | 4 | 47.38 |
| 4 | 1787806985 | 6 | 46.66 |
| 5 | 1787806987 | 8 | 46.37 |
| 6 | 1787806989 | 10 | 46.51 |
| 7 | 1787806991 | 12 | 27.76 |
| 8 | 1787806993 | 14 | 3.48 |
| 9 | 1787806995 | 16 | 3.23 |
| 10 | 1787806997 | 18 | 3.29 |
| 11 | 1787806999 | 20 | 3.30 |
| 12 | 1787807001 | 22 | 7.09 |
| 13 | 1787807003 | 24 | 3.70 |
| 14 | 1787807005 | 26 | 3.24 |
| 15 | 1787807007 | 28 | 15.70 |
| 16 | 1787807009 | 30 | 47.54 |
| 17 | 1787807011 | 32 | 4.76 |
| 18 | 1787807013 | 34 | 24.70 |
| 19 | 1787807015 | 36 | 48.11 |
| 20 | 1787807017 | 38 | 47.69 |
| 21 | 1787807019 | 40 | 47.44 |
| 22 | 1787807021 | 42 | 47.49 |
| 23 | 1787807023 | 44 | 47.55 |
| 24 | 1787807025 | 46 | 40.71 |
| 25 | 1787807027 | 48 | 46.87 |
| 26 | 1787807029 | 50 | 46.92 |
| 27 | 1787807031 | 52 | 46.76 |
| 28 | 1787807033 | 54 | 46.69 |
| 29 | 1787807035 | 56 | 46.87 |
| 30 | 1787807037 | 58 | 46.76 |
| 31 | 1787807039 | 60 | 47.19 |
| 32 | 1787807041 | 62 | 46.89 |
| 33 | 1787807043 | 64 | 47.35 |
| 34 | 1787807045 | 66 | 47.21 |
| 35 | 1787807047 | 68 | 46.83 |
| 36 | 1787807049 | 70 | 46.92 |
| 37 | 1787807051 | 72 | 46.79 |
| 38 | 1787807053 | 74 | 47.96 |
| 39 | 1787807055 | 76 | 46.76 |
| 40 | 1787807057 | 78 | 46.78 |
| 41 | 1787807059 | 80 | 46.57 |
| 42 | 1787807061 | 82 | 47.60 |
| 43 | 1787807063 | 84 | 34.36 |
| 44 | 1787807065 | 86 | 47.24 |
| 45 | 1787807067 | 88 | 47.26 |
| 46 | 1787807069 | 90 | 45.45 |
| 47 | 1787807071 | 92 | 47.67 |
| 48 | 1787807073 | 94 | 47.56 |
| 49 | 1787807075 | 96 | 47.53 |
| 50 | 1787807077 | 98 | 47.06 |
| 51 | 1787807079 | 100 | 46.92 |
| 52 | 1787807081 | 102 | 46.78 |
| 53 | 1787807083 | 104 | 47.02 |
| 54 | 1787807085 | 106 | 46.73 |
| 55 | 1787807087 | 108 | 47.18 |
| 56 | 1787807089 | 110 | 47.26 |
| 57 | 1787807091 | 112 | 47.02 |
| 58 | 1787807093 | 114 | 46.85 |
| 59 | 1787807095 | 116 | 46.87 |
| 60 | 1787807097 | 118 | 47.26 |
| 61 | 1787807099 | 120 | 47.11 |
| 62 | 1787807101 | 122 | 46.65 |
| 63 | 1787807103 | 124 | 47.46 |
| 64 | 1787807105 | 126 | 46.89 |
| 65 | 1787807107 | 128 | 47.77 |
| 66 | 1787807109 | 130 | 46.85 |
| 67 | 1787807111 | 132 | 46.73 |
| 68 | 1787807113 | 134 | 46.74 |
| 69 | 1787807115 | 136 | 47.30 |
| 70 | 1787807117 | 138 | 47.16 |
| 71 | 1787807119 | 140 | 47.47 |
| 72 | 1787807121 | 142 | 46.78 |
| 73 | 1787807123 | 144 | 47.95 |
| 74 | 1787807125 | 146 | 46.94 |
| 75 | 1787807127 | 148 | 47.96 |
| 76 | 1787807129 | 150 | 47.30 |
| 77 | 1787807131 | 152 | 47.23 |
| 78 | 1787807133 | 154 | 47.34 |
| 79 | 1787807135 | 156 | 47.43 |
| 80 | 1787807137 | 158 | 46.83 |
| 81 | 1787807140 | 161 | 46.66 |
| 82 | 1787807142 | 163 | 46.87 |
| 83 | 1787807144 | 165 | 46.90 |
| 84 | 1787807146 | 167 | 46.60 |
| 85 | 1787807148 | 169 | 46.83 |
| 86 | 1787807150 | 171 | 46.67 |
| 87 | 1787807152 | 173 | 46.62 |
| 88 | 1787807154 | 175 | 47.13 |
| 89 | 1787807156 | 177 | 47.06 |
| 90 | 1787807158 | 179 | 46.91 |
| 91 | 1787807160 | 181 | 46.77 |
| 92 | 1787807162 | 183 | 46.92 |
| 93 | 1787807164 | 185 | 47.32 |
| 94 | 1787807166 | 187 | 47.21 |
| 95 | 1787807168 | 189 | 47.15 |
| 96 | 1787807170 | 191 | 48.14 |
| 97 | 1787807172 | 193 | 47.06 |
| 98 | 1787807174 | 195 | 47.57 |
| 99 | 1787807176 | 197 | 47.09 |
| 100 | 1787807178 | 199 | 46.86 |
| 101 | 1787807180 | 201 | 46.66 |
| 102 | 1787807182 | 203 | 46.98 |
| 103 | 1787807184 | 205 | 47.32 |
| 104 | 1787807186 | 207 | 47.02 |
| 105 | 1787807188 | 209 | 49.49 |
| 106 | 1787807190 | 211 | 46.70 |
| 107 | 1787807192 | 213 | 47.06 |
| 108 | 1787807194 | 215 | 47.41 |
| 109 | 1787807196 | 217 | 47.07 |
| 110 | 1787807198 | 219 | 46.80 |
| 111 | 1787807200 | 221 | 46.80 |
| 112 | 1787807202 | 223 | 46.77 |
| 113 | 1787807204 | 225 | 46.98 |
| 114 | 1787807206 | 227 | 46.78 |
| 115 | 1787807208 | 229 | 47.52 |
| 116 | 1787807210 | 231 | 47.33 |
| 117 | 1787807212 | 233 | 46.76 |
| 118 | 1787807214 | 235 | 43.33 |

표본 118건 전량. 원문은 해당 `cpu_util.txt` 절에 있다.

### 3.5 벤치 호출 파라미터 (bench.log 첫 줄 Namespace)

| 파라미터 | smoke t1 — Qwen3-30B GPU-only | smoke t2 — Qwen3-30B KT hybrid | r1 — R1-0528 KT hybrid (AMXINT4) |
|---|---|---|---|
| `backend` | 'openai' | 'openai' | 'openai' |
| `base_url` | 'http://127.0.0.1:30000' | 'http://127.0.0.1:30000' | 'http://127.0.0.1:30000' |
| `endpoint` | '/v1/completions' | '/v1/completions' | '/v1/completions' |
| `model` | 'qwen30b' | 'qwen30b' | 'r1' |
| `dataset_name` | 'sonnet' | 'sonnet' | 'sonnet' |
| `num_prompts` | 128 | 128 | 32 |
| `sonnet_input_len` | 1024 | 1024 | 1024 |
| `sonnet_output_len` | 256 | 256 | 128 |
| `sonnet_prefix_len` | 200 | 200 | 200 |
| `max_concurrency` | 16 | 16 | 8 |
| `request_rate` | inf | inf | inf |
| `burstiness` | 1.0 | 1.0 | 1.0 |
| `temperature` | None | None | None |
| `ignore_eos` | False | False | False |
| `seed` | 42 | 42 | 42 |
| `percentile_metrics` | 'ttft,tpot,itl' | 'ttft,tpot,itl' | 'ttft,tpot,itl' |
| `metric_percentiles` | '50,95' | '50,95' | '50,95' |
| `num_warmups` | 0 | 0 | 0 |
| `tokenizer` | '/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169… | '/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169… | '/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6… |

규칙: `bench.log` 첫 줄의 `Namespace(...)` 를 정규식으로 뽑았다. 값은 원문 표기 그대로이며 70자를 넘으면 말줄임했다 (원문은 해당 절에 전문 수록).

세 셀 모두 `dataset_name='sonnet'`, `temperature=None` 이다. 후자는 `vllm bench serve` 가 더 이상 기본값으로 greedy 를 쓰지 않는다는 뜻이며, 같은 로그에 경고가 함께 기록되어 있다 — 품질 판별에 쓰인 greedy 출력은 벤치가 아니라 별도 단건 호출(`smoke.json`)이다.

---

# 원시 파일 — `eval/results/20260827_124509_tsk043_smoke_qwen30b/`

Qwen3-30B-A3B smoke — KT 스택 검증 (t1 GPU-only / t2 KT hybrid)

## 4. `eval/results/20260827_124509_tsk043_smoke_qwen30b/RUN.log`

- 바이트 **425** · 줄 **10** · SHA256 `936e38fdff28490ccd6e16a0182775a7f7a21a17493a09ca7b682c7c3336fe3d`
- 인코딩 utf-8

```text
== TSK_043 smoke start 20260827_124509 model=/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782 ==
-- t1_gpu_only --
SGL HEALTH OK 85s
-- t2_kt_hybrid --
SGL DIED 145s
t2 FAIL
== TSK_043 smoke done 125012 ==
== t2 retry (no cuda graph) 125132 BASE=/home/mystous/projects/vllm_hybrid/eval/results/20260827_124509_tsk043_smoke_qwen30b ==
SGL HEALTH OK 65s
== t2 retry done 125850 ==
```

## 5. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/.cpu_mon`

- 바이트 **8** · 줄 **1** · SHA256 `af590420629db1b3a94e558e7f73a2c109b5e217ff537af04915e9cb41caf84d`
- 인코딩 utf-8

```text
4133425
```

## 6. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/.gpu_mon`

- 바이트 **8** · 줄 **1** · SHA256 `b595785797104c7cde7b70ab80c00688059d3a3f9cca06312a3a194b38ea081b`
- 인코딩 utf-8

```text
4133423
```

## 7. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/bench.log`

- 바이트 **5,309** · 줄 **48** · SHA256 `a29a2fdab2bcbf3722911547e7b909fd5fe9e32c24d57a3d0f3d2d264d185bb5`
- 인코딩 utf-8 · CR 10건 치환

```text
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f5904502de0>, trust_remote_code=False, seed=42, num_prompts=128, dataset_name='sonnet', no_stream=False, dataset_path='/repo/benchmarks/sonnet.txt', no_oversample=False, skip_chat_template=False, enable_multimodal_chat=False, disable_shuffle=False, custom_output_len=256, custom_ensure_client_side_data=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=1024, sonnet_output_len=256, sonnet_prefix_len=200, sharegpt_output_len=None, timed_trace_chunk_hash_size=16, timed_trace_sec_multiplier=1, timed_trace_label_timestamp='timestamp', timed_trace_label_input_length='input_length', timed_trace_label_output_length='output_length', timed_trace_label_hash_ids='hash_ids', blazedit_min_distance=0.0, blazedit_max_distance=1.0, asr_max_audio_len_sec=inf, asr_min_audio_len_sec=0.0, random_input_len=1024, random_output_len=128, random_range_ratio='0.0', random_prefix_len=0, random_batch_size=1, no_reranker=False, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 1}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, bfcl_categories=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, speed_bench_dataset_subset='qualitative', speed_bench_output_len=4096, speed_bench_category=None, label=None, backend='openai', base_url='http://127.0.0.1:30000', host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=16, model='qwen30b', input_len=None, output_len=None, tokenizer='/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782', tokenizer_mode='auto', use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, probe_request_rate=0.0, disable_tqdm=False, num_warmups=0, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, self_timed=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='50,95', goodput=None, request_id_prefix='bench-d757fa48-', top_p=None, top_k=None, min_p=None, temperature=None, frequency_penalty=None, presence_penalty=None, repetition_penalty=None, served_model_name=None, lora_modules=None, lora_assignment='random', ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=0, chat_template_kwargs=None, extra_body=None, skip_tokenizer_init=False, insecure=False, plot_timeline=False, timeline_itl_thresholds='25,50', plot_dataset_stats=False)
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16

⏎   0%|          | 0/128 [00:00<?, ?it/s]
⏎   1%|          | 1/128 [00:03<08:23,  3.97s/it]
⏎  13%|█▎        | 17/128 [00:06<00:35,  3.09it/s]
⏎  26%|██▌       | 33/128 [00:09<00:22,  4.30it/s]
⏎  38%|███▊      | 49/128 [00:11<00:15,  4.96it/s]
⏎  51%|█████     | 65/128 [00:14<00:11,  5.34it/s]
⏎  63%|██████▎   | 81/128 [00:17<00:08,  5.58it/s]
⏎  76%|███████▌  | 97/128 [00:19<00:05,  5.74it/s]
⏎  88%|████████▊ | 113/128 [00:22<00:02,  5.85it/s]
⏎ 100%|██████████| 128/128 [00:22<00:00,  5.71it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  22.42     
Total input tokens:                      129773    
Total generated tokens:                  32768     
Request throughput (req/s):              5.71      
Output token throughput (tok/s):         1461.77   
Peak output token throughput (tok/s):    1719.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          7250.88   
---------------Time to First Token----------------
Mean TTFT (ms):                          342.89    
Median TTFT (ms):                        179.28    
P50 TTFT (ms):                           179.28    
P95 TTFT (ms):                           1533.03   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.64      
Median TPOT (ms):                        9.66      
P50 TPOT (ms):                           9.66      
P95 TPOT (ms):                           10.06     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.64      
Median ITL (ms):                         9.53      
P50 ITL (ms):                            9.53      
P95 ITL (ms):                            9.94      
==================================================
```

## 8. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/cpu_util.txt`

- 바이트 **399** · 줄 **19** · SHA256 `1eaf41a2c6ef9715fe11b6e719999e9260c2d5c06fba52cf59cb436dfcbbdcbe`
- 인코딩 utf-8

```text
1787802399 busy=3.77
1787802401 busy=1.79
1787802403 busy=1.76
1787802405 busy=1.69
1787802407 busy=5.71
1787802409 busy=2.01
1787802411 busy=1.87
1787802413 busy=1.73
1787802415 busy=1.33
1787802417 busy=1.76
1787802419 busy=1.46
1787802421 busy=1.39
1787802423 busy=1.85
1787802425 busy=1.67
1787802427 busy=1.64
1787802429 busy=1.33
1787802431 busy=1.42
1787802433 busy=1.60
1787802435 busy=1.78
```

## 9. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/gpu_util.csv`

- 바이트 **1,519** · 줄 **64** · SHA256 `43b943f1432773e7632e67f8c9d8c3c5971ac2b72f6fc1fd78a89195ff55a04d`
- 인코딩 utf-8

```text
0, 4 %, 70320 MiB, 129.02 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.10 W
3, 0 %, 4 MiB, 69.29 W
4, 0 %, 4 MiB, 69.73 W
5, 0 %, 4 MiB, 68.85 W
6, 0 %, 4 MiB, 70.23 W
7, 0 %, 4 MiB, 70.83 W
0, 0 %, 70320 MiB, 115.28 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.29 W
4, 0 %, 4 MiB, 69.70 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.84 W
0, 0 %, 70320 MiB, 115.23 W
1, 0 %, 4 MiB, 68.66 W
2, 0 %, 4 MiB, 69.22 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.70 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.15 W
7, 0 %, 4 MiB, 70.84 W
0, 0 %, 70320 MiB, 115.22 W
1, 0 %, 4 MiB, 68.74 W
2, 0 %, 4 MiB, 69.18 W
3, 0 %, 4 MiB, 69.28 W
4, 0 %, 4 MiB, 69.64 W
5, 0 %, 4 MiB, 68.86 W
6, 0 %, 4 MiB, 70.15 W
7, 0 %, 4 MiB, 70.86 W
0, 100 %, 70364 MiB, 421.56 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.12 W
3, 0 %, 4 MiB, 69.28 W
4, 0 %, 4 MiB, 69.70 W
5, 0 %, 4 MiB, 68.83 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.83 W
0, 100 %, 70364 MiB, 432.81 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.06 W
3, 0 %, 4 MiB, 69.30 W
4, 0 %, 4 MiB, 69.65 W
5, 0 %, 4 MiB, 68.76 W
6, 0 %, 4 MiB, 70.24 W
7, 0 %, 4 MiB, 70.81 W
0, 86 %, 70364 MiB, 415.49 W
1, 0 %, 4 MiB, 68.72 W
2, 0 %, 4 MiB, 69.08 W
3, 0 %, 4 MiB, 69.29 W
4, 0 %, 4 MiB, 69.63 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.81 W
0, 100 %, 70364 MiB, 382.89 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.31 W
4, 0 %, 4 MiB, 69.64 W
5, 0 %, 4 MiB, 68.83 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.88 W
```

## 10. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/server.log`

- 바이트 **70,056** · 줄 **464** · SHA256 `03644dbe5399a3e0cf676259b737d3fb6741eb0690c284ba43544df2194bb39d`
- 인코딩 utf-8 · CR 201건 치환

```text
/sgl-workspace/sglang/python/sglang/launch_server.py:57: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
  Example: sglang serve --model-path <model> [options]
  warnings.warn(
[2026-08-27 03:45:24] server_args=ServerArgs(model_path='/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782', tokenizer_path='/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782', tokenizer_mode='auto', tokenizer_backend='huggingface', tokenizer_worker_num=1, detokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', model_config_parser='auto', json_model_override_args='{}', dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, enable_tf32_matmul=False, mem_fraction_static=0.83, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, retraction_policy='length', schedule_conservativeness=1.0, page_size=1, c128_page_size=16, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', prefill_only_disable_kv_cache=False, disable_radix_cache=False, enable_page_major_kv_layout=False, enable_unified_memory=False, disable_chunked_prefix_cache=False, disable_overlap_schedule=False, num_continuous_decode_steps=1, scheduler_recv_interval=1, enable_mixed_chunk=False, nccl_port=None, dist_timeout=None, dist_init_addr=None, nnodes=1, node_rank=0, tp_size=1, dcp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dwdp_size=1, dcp_comm_backend='ag_rs', dcp_replicate_q_proj=None, enable_prefill_cp=False, cp_strategy=None, enable_dsa_cache_layer_split=False, enable_dsa_prefill_context_parallel=False, dsa_prefill_cp_mode='round-robin-split', enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_cp_decode_attn_tp=False, enable_dp_attention=False, enable_dp_attention_local_control_broadcast=False, enable_dp_lm_head=False, enable_tp_lm_head_all_to_all=False, enable_attn_tp_input_scattered=False, disable_attn_tp_gather=False, enable_p2p_check=False, device='cuda', base_gpu_id=0, gpu_id_step=1, random_seed=218841610, mlx_enable_sampling=False, watchdog_timeout=300, soft_watchdog_timeout=None, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, numa_node=None, gc_threshold=None, host='127.0.0.1', port=30000, fastapi_root_path='', smg_grpc_mode=False, grpc_mode=False, grpc_port=None, sidecar=None, sidecar_args=None, skip_server_warmup=False, warmups=None, enable_http2=False, http2_max_concurrent_streams=200, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, api_key=None, admin_api_key=None, served_model_name='qwen30b', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, default_chat_template_kwargs=None, strip_thinking_cache=False, enable_strict_thinking=False, tool_call_parser=None, tool_server=None, sampling_defaults='model', asr_max_buffer_seconds=60, asr_max_concurrent_sessions=32, preferred_sampling_params=None, allow_auto_truncate=False, stream_interval=1, batch_notify_size=16, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, enable_session_radix_cache=False, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, smg_http_sidecar_port=None, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, load_snapshot_publish_interval=15, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_forward_pass_metrics=False, forward_pass_metrics_worker_id='', forward_pass_metrics_ipc_name=None, enable_trace=False, trace_modules='request', otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, stat_loggers=None, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', radix_cache_backend=None, mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', bf16_gemm_backend='auto', dsa_prefill_backend=None, dsa_decode_backend=None, dsa_paged_mqa_logits_backend='auto', dsa_topk_backend='sgl-kernel', disable_flashinfer_autotune=False, flashinfer_autotune_skip_ops=None, mamba_backend='triton', cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend='full', max_bs=256, bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], tc_compiler='eager', full_prefill_max_req=None, full_prefill_prefix_chunk_tokens=None), prefill=PhaseConfig(backend='breakable', max_bs=8192, bs=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], tc_compiler='eager', full_prefill_max_req=None, full_prefill_prefix_chunk_tokens=None)), cuda_graph_backend_decode=None, cuda_graph_backend_prefill=None, cuda_graph_max_bs_decode=None, cuda_graph_max_bs_prefill=None, cuda_graph_bs_decode=None, cuda_graph_bs_prefill=None, cuda_graph_tc_compiler=None, disable_prefill_cuda_graph=False, disable_decode_cuda_graph=False, disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, debug_cuda_graph=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, flashinfer_mla_disable_ragged=False, enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_deepseek_v4_fp4_indexer=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, enable_scattered_sconv=False, pre_warm_nccl=False, enable_quant_communications=False, enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, flashinfer_allreduce_fusion_backend=None, enable_aiter_allreduce_fusion=False, enable_torch_compile=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_dflash_block_size=None, speculative_dspark_block_size=None, speculative_dspark_sps_table_path=None, speculative_dspark_confidence_sts_path=None, speculative_dspark_align_verify_tokens_to_graph_tier=False, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_use_rejection_sampling=False, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_draft_kv_cache_dtype=None, speculative_draft_window_size=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, _speculative_draft_quantization_explicitly_set=False, speculative_skip_dp_mlp_sync=False, enable_multi_layer_eagle=False, speculative_adaptive=False, speculative_adaptive_config=None, decoupled_spec_bind_endpoint=None, decoupled_spec_connect_endpoints=None, decoupled_spec_rank=None, decoupled_spec_role='null', spec_trace_dir=None, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, speculative_ngram_external_corpus_path=None, speculative_ngram_external_sam_budget=0, speculative_ngram_external_corpus_max_tokens=10000000, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', deepep_mode='auto', fuseep_mode=2, deepep_dispatcher_output_dtype='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, expert_balancedness_report_mode='off', deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, enable_waterfill=False, ep_join_mode=None, ep_join_rank_offset=0, elastic_ep_initial_size=None, max_ep_size=None, elastic_ep_scale_timeout=600, elastic_ep_rejoin=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, disable_shared_experts_fusion=False, enforce_shared_experts_fusion=False, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_max_states_per_path=-1, enable_mamba_cache_stochastic_rounding=False, mamba_cache_philox_rounds=0, mamba_full_memory_ratio=0.9, mamba_radix_cache_strategy='auto', uses_mamba_radix_cache=False, mamba_track_interval=256, enable_int8_mamba_checkpoint=False, int8_mamba_ckpt_size=None, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, linear_attn_verify_backend=None, enable_linear_replayssm=False, linear_replayssm_cache_len=16, enable_linear_replayssm_spec=False, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='page_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='timeout', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, mm_processor_worker_num=0, mm_io_worker_num=0, allowed_media_domains=[], media_url_max_file_size_mb=64, mm_preprocess_cache_size_mb=None, trust_mm_content_hashes=False, limit_mm_data_per_request=None, enable_mm_global_cache=False, image_processor_backend='auto', mm_global_cache_backend='mooncake', disable_fast_image_processor=False, mm_feature_transport='cpu', keep_mm_feature_on_device=False, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, lora_use_virtual_experts=False, lora_strict_loading=False, lora_drain_wait_threshold=0.0, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', enable_lmcache=False, lmcache_config_file=None, enable_flexkv=False, flexkv_config_file=None, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, dllm_fdfo=True, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_radix_cache=False, disaggregation_decode_enable_offload_kvcache=False, disaggregation_decode_retraction_backup=None, num_reserved_decode_tokens=512, disaggregation_decode_extra_slots=None, disaggregation_decode_polling_interval=1, optimistic_prefill_attempts=0, encoder_only=False, language_only=False, language_model_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], encoder_bootstrap_port=8997, encoder_register_urls=[], enable_adaptive_dispatch_to_encoder=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, startup_weight_load_mode='serial', custom_weight_loader=[], weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False, weight_loader_prefetch_num_threads=4, weight_loader_drop_cache_after_load=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, download_dir=None, model_checksum=None, delete_ckpt_after_loading=False, decrypted_config_file=None, decrypted_draft_config_file=None, checkpoint_engine_wait_weights_before_ready=False, enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, prefill_delayer_queue_min_ratio=None, prefill_delayer_max_delay_ms=None, min_free_slots_delay=None, enable_deterministic_inference=False, rl_on_policy_target=None, kv_canary='none', kv_canary_real_data='none', kv_canary_sweep_interval=0, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, enable_custom_logit_processor=False, enable_return_hidden_states=False, return_hidden_states_mode=None, enable_return_routed_experts=False, enable_return_indexer_topk=False, disable_outlines_disk_cache=False, enable_mis=False, weight_cache_mode='off', weight_cache_socket=None, weight_cache_timeout=1800, forward_hooks=None, msprobe_dump_config=None)
[2026-08-27 03:45:26] Using default HuggingFace chat template with detected content format: string
[2026-08-27 03:45:27] Auto-detected template features: reasoning_config=ReasoningToggleConfig(toggle_param='enable_thinking', default_enabled=True, special_case=None, effort_kwarg=None), reasoning_parser=qwen3, tool_call_parser=qwen
[2026-08-27 03:45:39] Init torch distributed begin.
[2026-08-27 03:45:39] Init torch distributed ends. elapsed=0.05 s, mem usage=0.09 GB
[2026-08-27 03:45:41] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 03:45:41] Load weight begin. avail mem=78.48 GB
[2026-08-27 03:45:41] Detected fp8 checkpoint.
[2026-08-27 03:45:41] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=Fp8MoEMethod).

⏎ Multi-thread loading shards:   0% Completed | 0/7 [00:00<?, ?it/s]
⏎ Multi-thread loading shards:  14% Completed | 1/7 [00:02<00:17,  2.84s/it]
⏎ Multi-thread loading shards:  29% Completed | 2/7 [00:04<00:10,  2.11s/it]
⏎ Multi-thread loading shards:  43% Completed | 3/7 [00:06<00:07,  1.89s/it]
⏎ Multi-thread loading shards:  57% Completed | 4/7 [00:07<00:05,  1.72s/it]
⏎ Multi-thread loading shards:  71% Completed | 5/7 [00:09<00:03,  1.69s/it]
⏎ Multi-thread loading shards:  86% Completed | 6/7 [00:10<00:01,  1.66s/it]
⏎ Multi-thread loading shards: 100% Completed | 7/7 [00:11<00:00,  1.38s/it]
⏎ Multi-thread loading shards: 100% Completed | 7/7 [00:11<00:00,  1.65s/it]
[2026-08-27 03:45:53] Load weight end. elapsed=12.02 s, type=Qwen3MoeForCausalLM, quant=fp8, fmt=e4m3, avail mem=49.33 GB, mem usage=29.15 GB.
[2026-08-27 03:45:54] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 392920, K size: 17.99 GB, V size: 17.99 GB
[2026-08-27 03:45:54] Memory pool end. avail mem=12.70 GB
[2026-08-27 03:45:55] Capture target prefill CUDA graph begin. backend=breakable, num_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], avail mem=12.59 GB

⏎   0%|          | 0/58 [00:00<?, ?it/s]
⏎ Capturing num tokens (num_tokens=8192 avail_mem=12.59 GB):   0%|          | 0/58 [00:00<?, ?it/s][2026-08-27 03:45:56] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=5120,K=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 03:45:56] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2048,K=4096,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 03:45:56] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 03:45:56] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton

⏎ Capturing num tokens (num_tokens=8192 avail_mem=12.59 GB):   2%|▏         | 1/58 [00:01<01:27,  1.54s/it]
⏎ Capturing num tokens (num_tokens=7680 avail_mem=10.30 GB):   2%|▏         | 1/58 [00:01<01:27,  1.54s/it]
⏎ Capturing num tokens (num_tokens=7680 avail_mem=10.30 GB):   3%|▎         | 2/58 [00:02<00:53,  1.04it/s]
⏎ Capturing num tokens (num_tokens=7168 avail_mem=10.30 GB):   3%|▎         | 2/58 [00:02<00:53,  1.04it/s]
⏎ Capturing num tokens (num_tokens=7168 avail_mem=10.30 GB):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]
⏎ Capturing num tokens (num_tokens=6656 avail_mem=10.29 GB):   5%|▌         | 3/58 [00:02<00:41,  1.32it/s]
⏎ Capturing num tokens (num_tokens=6656 avail_mem=10.29 GB):   7%|▋         | 4/58 [00:03<00:34,  1.55it/s]
⏎ Capturing num tokens (num_tokens=6144 avail_mem=10.28 GB):   7%|▋         | 4/58 [00:03<00:34,  1.55it/s]
⏎ Capturing num tokens (num_tokens=6144 avail_mem=10.28 GB):   9%|▊         | 5/58 [00:03<00:30,  1.75it/s]
⏎ Capturing num tokens (num_tokens=5632 avail_mem=10.27 GB):   9%|▊         | 5/58 [00:03<00:30,  1.75it/s]
⏎ Capturing num tokens (num_tokens=5632 avail_mem=10.27 GB):  10%|█         | 6/58 [00:03<00:26,  1.98it/s]
⏎ Capturing num tokens (num_tokens=5120 avail_mem=10.27 GB):  10%|█         | 6/58 [00:03<00:26,  1.98it/s]
⏎ Capturing num tokens (num_tokens=5120 avail_mem=10.27 GB):  12%|█▏        | 7/58 [00:04<00:23,  2.20it/s]
⏎ Capturing num tokens (num_tokens=4608 avail_mem=10.26 GB):  12%|█▏        | 7/58 [00:04<00:23,  2.20it/s]
⏎ Capturing num tokens (num_tokens=4608 avail_mem=10.26 GB):  14%|█▍        | 8/58 [00:04<00:20,  2.39it/s]
⏎ Capturing num tokens (num_tokens=4096 avail_mem=10.25 GB):  14%|█▍        | 8/58 [00:04<00:20,  2.39it/s]
⏎ Capturing num tokens (num_tokens=4096 avail_mem=10.25 GB):  16%|█▌        | 9/58 [00:04<00:19,  2.53it/s]
⏎ Capturing num tokens (num_tokens=3840 avail_mem=10.24 GB):  16%|█▌        | 9/58 [00:04<00:19,  2.53it/s]
⏎ Capturing num tokens (num_tokens=3840 avail_mem=10.24 GB):  17%|█▋        | 10/58 [00:05<00:18,  2.63it/s]
⏎ Capturing num tokens (num_tokens=3584 avail_mem=10.24 GB):  17%|█▋        | 10/58 [00:05<00:18,  2.63it/s]
⏎ Capturing num tokens (num_tokens=3584 avail_mem=10.24 GB):  19%|█▉        | 11/58 [00:05<00:17,  2.73it/s]
⏎ Capturing num tokens (num_tokens=3328 avail_mem=10.23 GB):  19%|█▉        | 11/58 [00:05<00:17,  2.73it/s]
⏎ Capturing num tokens (num_tokens=3328 avail_mem=10.23 GB):  21%|██        | 12/58 [00:05<00:16,  2.81it/s]
⏎ Capturing num tokens (num_tokens=3072 avail_mem=10.22 GB):  21%|██        | 12/58 [00:05<00:16,  2.81it/s]
⏎ Capturing num tokens (num_tokens=3072 avail_mem=10.22 GB):  22%|██▏       | 13/58 [00:06<00:15,  2.84it/s]
⏎ Capturing num tokens (num_tokens=2816 avail_mem=10.21 GB):  22%|██▏       | 13/58 [00:06<00:15,  2.84it/s]
⏎ Capturing num tokens (num_tokens=2816 avail_mem=10.21 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.87it/s]
⏎ Capturing num tokens (num_tokens=2560 avail_mem=10.21 GB):  24%|██▍       | 14/58 [00:06<00:15,  2.87it/s]
⏎ Capturing num tokens (num_tokens=2560 avail_mem=10.21 GB):  26%|██▌       | 15/58 [00:06<00:14,  2.91it/s]
⏎ Capturing num tokens (num_tokens=2304 avail_mem=10.20 GB):  26%|██▌       | 15/58 [00:06<00:14,  2.91it/s]
⏎ Capturing num tokens (num_tokens=2304 avail_mem=10.20 GB):  28%|██▊       | 16/58 [00:07<00:14,  2.93it/s]
⏎ Capturing num tokens (num_tokens=2048 avail_mem=10.19 GB):  28%|██▊       | 16/58 [00:07<00:14,  2.93it/s]
⏎ Capturing num tokens (num_tokens=2048 avail_mem=10.19 GB):  29%|██▉       | 17/58 [00:07<00:13,  2.94it/s]
⏎ Capturing num tokens (num_tokens=1792 avail_mem=10.18 GB):  29%|██▉       | 17/58 [00:07<00:13,  2.94it/s]
⏎ Capturing num tokens (num_tokens=1792 avail_mem=10.18 GB):  31%|███       | 18/58 [00:07<00:13,  2.93it/s]
⏎ Capturing num tokens (num_tokens=1536 avail_mem=10.18 GB):  31%|███       | 18/58 [00:07<00:13,  2.93it/s]
⏎ Capturing num tokens (num_tokens=1536 avail_mem=10.18 GB):  33%|███▎      | 19/58 [00:08<00:13,  2.94it/s]
⏎ Capturing num tokens (num_tokens=1280 avail_mem=10.17 GB):  33%|███▎      | 19/58 [00:08<00:13,  2.94it/s]
⏎ Capturing num tokens (num_tokens=1280 avail_mem=10.17 GB):  34%|███▍      | 20/58 [00:08<00:12,  2.95it/s]
⏎ Capturing num tokens (num_tokens=1024 avail_mem=10.16 GB):  34%|███▍      | 20/58 [00:08<00:12,  2.95it/s]
⏎ Capturing num tokens (num_tokens=1024 avail_mem=10.16 GB):  36%|███▌      | 21/58 [00:08<00:12,  2.95it/s]
⏎ Capturing num tokens (num_tokens=960 avail_mem=10.15 GB):  36%|███▌      | 21/58 [00:08<00:12,  2.95it/s] 
⏎ Capturing num tokens (num_tokens=960 avail_mem=10.15 GB):  38%|███▊      | 22/58 [00:09<00:12,  2.93it/s]
⏎ Capturing num tokens (num_tokens=896 avail_mem=10.15 GB):  38%|███▊      | 22/58 [00:09<00:12,  2.93it/s]
⏎ Capturing num tokens (num_tokens=896 avail_mem=10.15 GB):  40%|███▉      | 23/58 [00:09<00:11,  2.95it/s]
⏎ Capturing num tokens (num_tokens=832 avail_mem=10.14 GB):  40%|███▉      | 23/58 [00:09<00:11,  2.95it/s]
⏎ Capturing num tokens (num_tokens=832 avail_mem=10.14 GB):  41%|████▏     | 24/58 [00:10<00:11,  2.96it/s]
⏎ Capturing num tokens (num_tokens=768 avail_mem=10.13 GB):  41%|████▏     | 24/58 [00:10<00:11,  2.96it/s]
⏎ Capturing num tokens (num_tokens=768 avail_mem=10.13 GB):  43%|████▎     | 25/58 [00:10<00:11,  2.97it/s]
⏎ Capturing num tokens (num_tokens=704 avail_mem=10.12 GB):  43%|████▎     | 25/58 [00:10<00:11,  2.97it/s]
⏎ Capturing num tokens (num_tokens=704 avail_mem=10.12 GB):  45%|████▍     | 26/58 [00:10<00:10,  2.97it/s]
⏎ Capturing num tokens (num_tokens=640 avail_mem=10.11 GB):  45%|████▍     | 26/58 [00:10<00:10,  2.97it/s]
⏎ Capturing num tokens (num_tokens=640 avail_mem=10.11 GB):  47%|████▋     | 27/58 [00:11<00:10,  2.98it/s]
⏎ Capturing num tokens (num_tokens=576 avail_mem=10.11 GB):  47%|████▋     | 27/58 [00:11<00:10,  2.98it/s]
⏎ Capturing num tokens (num_tokens=576 avail_mem=10.11 GB):  48%|████▊     | 28/58 [00:11<00:09,  3.03it/s]
⏎ Capturing num tokens (num_tokens=512 avail_mem=10.10 GB):  48%|████▊     | 28/58 [00:11<00:09,  3.03it/s]
⏎ Capturing num tokens (num_tokens=512 avail_mem=10.10 GB):  50%|█████     | 29/58 [00:11<00:09,  3.02it/s]
⏎ Capturing num tokens (num_tokens=480 avail_mem=10.10 GB):  50%|█████     | 29/58 [00:11<00:09,  3.02it/s]
⏎ Capturing num tokens (num_tokens=480 avail_mem=10.10 GB):  52%|█████▏    | 30/58 [00:11<00:09,  3.02it/s]
⏎ Capturing num tokens (num_tokens=448 avail_mem=10.09 GB):  52%|█████▏    | 30/58 [00:11<00:09,  3.02it/s]
⏎ Capturing num tokens (num_tokens=448 avail_mem=10.09 GB):  53%|█████▎    | 31/58 [00:12<00:08,  3.02it/s]
⏎ Capturing num tokens (num_tokens=416 avail_mem=10.08 GB):  53%|█████▎    | 31/58 [00:12<00:08,  3.02it/s]
⏎ Capturing num tokens (num_tokens=416 avail_mem=10.08 GB):  55%|█████▌    | 32/58 [00:12<00:08,  3.01it/s]
⏎ Capturing num tokens (num_tokens=384 avail_mem=10.07 GB):  55%|█████▌    | 32/58 [00:12<00:08,  3.01it/s]
⏎ Capturing num tokens (num_tokens=384 avail_mem=10.07 GB):  57%|█████▋    | 33/58 [00:12<00:08,  3.00it/s]
⏎ Capturing num tokens (num_tokens=352 avail_mem=10.06 GB):  57%|█████▋    | 33/58 [00:12<00:08,  3.00it/s]
⏎ Capturing num tokens (num_tokens=352 avail_mem=10.06 GB):  59%|█████▊    | 34/58 [00:13<00:07,  3.01it/s]
⏎ Capturing num tokens (num_tokens=320 avail_mem=10.06 GB):  59%|█████▊    | 34/58 [00:13<00:07,  3.01it/s]
⏎ Capturing num tokens (num_tokens=320 avail_mem=10.06 GB):  60%|██████    | 35/58 [00:13<00:07,  3.01it/s]
⏎ Capturing num tokens (num_tokens=288 avail_mem=10.05 GB):  60%|██████    | 35/58 [00:13<00:07,  3.01it/s]
⏎ Capturing num tokens (num_tokens=288 avail_mem=10.05 GB):  62%|██████▏   | 36/58 [00:13<00:07,  3.01it/s]
⏎ Capturing num tokens (num_tokens=256 avail_mem=10.04 GB):  62%|██████▏   | 36/58 [00:13<00:07,  3.01it/s]
⏎ Capturing num tokens (num_tokens=256 avail_mem=10.04 GB):  64%|██████▍   | 37/58 [00:14<00:06,  3.01it/s]
⏎ Capturing num tokens (num_tokens=240 avail_mem=10.03 GB):  64%|██████▍   | 37/58 [00:14<00:06,  3.01it/s]
⏎ Capturing num tokens (num_tokens=240 avail_mem=10.03 GB):  66%|██████▌   | 38/58 [00:14<00:06,  3.01it/s]
⏎ Capturing num tokens (num_tokens=224 avail_mem=10.03 GB):  66%|██████▌   | 38/58 [00:14<00:06,  3.01it/s]
⏎ Capturing num tokens (num_tokens=224 avail_mem=10.03 GB):  67%|██████▋   | 39/58 [00:14<00:06,  3.00it/s]
⏎ Capturing num tokens (num_tokens=208 avail_mem=10.02 GB):  67%|██████▋   | 39/58 [00:14<00:06,  3.00it/s]
⏎ Capturing num tokens (num_tokens=208 avail_mem=10.02 GB):  69%|██████▉   | 40/58 [00:15<00:05,  3.00it/s]
⏎ Capturing num tokens (num_tokens=192 avail_mem=10.01 GB):  69%|██████▉   | 40/58 [00:15<00:05,  3.00it/s]
⏎ Capturing num tokens (num_tokens=192 avail_mem=10.01 GB):  71%|███████   | 41/58 [00:15<00:05,  2.99it/s]
⏎ Capturing num tokens (num_tokens=176 avail_mem=10.00 GB):  71%|███████   | 41/58 [00:15<00:05,  2.99it/s]
⏎ Capturing num tokens (num_tokens=176 avail_mem=10.00 GB):  72%|███████▏  | 42/58 [00:16<00:05,  2.97it/s]
⏎ Capturing num tokens (num_tokens=160 avail_mem=9.99 GB):  72%|███████▏  | 42/58 [00:16<00:05,  2.97it/s] 
⏎ Capturing num tokens (num_tokens=160 avail_mem=9.99 GB):  74%|███████▍  | 43/58 [00:16<00:05,  2.97it/s]
⏎ Capturing num tokens (num_tokens=144 avail_mem=9.98 GB):  74%|███████▍  | 43/58 [00:16<00:05,  2.97it/s]
⏎ Capturing num tokens (num_tokens=144 avail_mem=9.98 GB):  76%|███████▌  | 44/58 [00:16<00:04,  2.98it/s]
⏎ Capturing num tokens (num_tokens=128 avail_mem=9.98 GB):  76%|███████▌  | 44/58 [00:16<00:04,  2.98it/s]
⏎ Capturing num tokens (num_tokens=128 avail_mem=9.98 GB):  78%|███████▊  | 45/58 [00:17<00:04,  2.94it/s]
⏎ Capturing num tokens (num_tokens=112 avail_mem=9.97 GB):  78%|███████▊  | 45/58 [00:17<00:04,  2.94it/s]
⏎ Capturing num tokens (num_tokens=112 avail_mem=9.97 GB):  79%|███████▉  | 46/58 [00:17<00:04,  2.94it/s]
⏎ Capturing num tokens (num_tokens=96 avail_mem=9.96 GB):  79%|███████▉  | 46/58 [00:17<00:04,  2.94it/s] 
⏎ Capturing num tokens (num_tokens=96 avail_mem=9.96 GB):  81%|████████  | 47/58 [00:17<00:03,  2.96it/s]
⏎ Capturing num tokens (num_tokens=80 avail_mem=9.95 GB):  81%|████████  | 47/58 [00:17<00:03,  2.96it/s]
⏎ Capturing num tokens (num_tokens=80 avail_mem=9.95 GB):  83%|████████▎ | 48/58 [00:18<00:03,  2.97it/s]
⏎ Capturing num tokens (num_tokens=64 avail_mem=9.95 GB):  83%|████████▎ | 48/58 [00:18<00:03,  2.97it/s]
⏎ Capturing num tokens (num_tokens=64 avail_mem=9.95 GB):  84%|████████▍ | 49/58 [00:18<00:03,  3.00it/s]
⏎ Capturing num tokens (num_tokens=48 avail_mem=9.94 GB):  84%|████████▍ | 49/58 [00:18<00:03,  3.00it/s]
⏎ Capturing num tokens (num_tokens=48 avail_mem=9.94 GB):  86%|████████▌ | 50/58 [00:18<00:02,  3.01it/s]
⏎ Capturing num tokens (num_tokens=32 avail_mem=9.93 GB):  86%|████████▌ | 50/58 [00:18<00:02,  3.01it/s]
⏎ Capturing num tokens (num_tokens=32 avail_mem=9.93 GB):  88%|████████▊ | 51/58 [00:19<00:03,  2.27it/s]
⏎ Capturing num tokens (num_tokens=28 avail_mem=9.93 GB):  88%|████████▊ | 51/58 [00:19<00:03,  2.27it/s]
⏎ Capturing num tokens (num_tokens=28 avail_mem=9.93 GB):  90%|████████▉ | 52/58 [00:19<00:02,  2.18it/s]
⏎ Capturing num tokens (num_tokens=24 avail_mem=9.92 GB):  90%|████████▉ | 52/58 [00:19<00:02,  2.18it/s]
⏎ Capturing num tokens (num_tokens=24 avail_mem=9.92 GB):  91%|█████████▏| 53/58 [00:20<00:02,  2.33it/s]
⏎ Capturing num tokens (num_tokens=20 avail_mem=9.91 GB):  91%|█████████▏| 53/58 [00:20<00:02,  2.33it/s]
⏎ Capturing num tokens (num_tokens=20 avail_mem=9.91 GB):  93%|█████████▎| 54/58 [00:20<00:01,  2.46it/s]
⏎ Capturing num tokens (num_tokens=16 avail_mem=9.90 GB):  93%|█████████▎| 54/58 [00:20<00:01,  2.46it/s]
⏎ Capturing num tokens (num_tokens=16 avail_mem=9.90 GB):  95%|█████████▍| 55/58 [00:21<00:01,  2.43it/s]
⏎ Capturing num tokens (num_tokens=12 avail_mem=9.90 GB):  95%|█████████▍| 55/58 [00:21<00:01,  2.43it/s]
⏎ Capturing num tokens (num_tokens=12 avail_mem=9.90 GB):  97%|█████████▋| 56/58 [00:21<00:00,  2.53it/s]
⏎ Capturing num tokens (num_tokens=8 avail_mem=9.89 GB):  97%|█████████▋| 56/58 [00:21<00:00,  2.53it/s] 
⏎ Capturing num tokens (num_tokens=8 avail_mem=9.89 GB):  98%|█████████▊| 57/58 [00:21<00:00,  2.59it/s]
⏎ Capturing num tokens (num_tokens=4 avail_mem=9.88 GB):  98%|█████████▊| 57/58 [00:21<00:00,  2.59it/s]
⏎ Capturing num tokens (num_tokens=4 avail_mem=9.88 GB): 100%|██████████| 58/58 [00:22<00:00,  2.61it/s]
⏎ Capturing num tokens (num_tokens=4 avail_mem=9.88 GB): 100%|██████████| 58/58 [00:22<00:00,  2.62it/s]
[2026-08-27 03:46:18] Capture target prefill CUDA graph end. elapsed=22.96 s, mem usage=1.71 GB, avail mem=10.88 GB.
[2026-08-27 03:46:18] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], avail mem=10.88 GB

⏎   0%|          | 0/36 [00:00<?, ?it/s]
⏎ Capturing batches (bs=256 avail_mem=10.77 GB):   0%|          | 0/36 [00:00<?, ?it/s]
⏎ Capturing batches (bs=256 avail_mem=10.77 GB):   3%|▎         | 1/36 [00:00<00:11,  3.10it/s]
⏎ Capturing batches (bs=248 avail_mem=10.64 GB):   3%|▎         | 1/36 [00:00<00:11,  3.10it/s]
⏎ Capturing batches (bs=248 avail_mem=10.64 GB):   6%|▌         | 2/36 [00:00<00:10,  3.16it/s]
⏎ Capturing batches (bs=240 avail_mem=10.64 GB):   6%|▌         | 2/36 [00:00<00:10,  3.16it/s]
⏎ Capturing batches (bs=240 avail_mem=10.64 GB):   8%|▊         | 3/36 [00:00<00:10,  3.20it/s]
⏎ Capturing batches (bs=232 avail_mem=10.63 GB):   8%|▊         | 3/36 [00:00<00:10,  3.20it/s]
⏎ Capturing batches (bs=232 avail_mem=10.63 GB):  11%|█         | 4/36 [00:01<00:10,  3.20it/s]
⏎ Capturing batches (bs=224 avail_mem=10.63 GB):  11%|█         | 4/36 [00:01<00:10,  3.20it/s]
⏎ Capturing batches (bs=224 avail_mem=10.63 GB):  14%|█▍        | 5/36 [00:01<00:09,  3.22it/s]
⏎ Capturing batches (bs=216 avail_mem=10.63 GB):  14%|█▍        | 5/36 [00:01<00:09,  3.22it/s]
⏎ Capturing batches (bs=216 avail_mem=10.63 GB):  17%|█▋        | 6/36 [00:01<00:09,  3.19it/s]
⏎ Capturing batches (bs=208 avail_mem=10.62 GB):  17%|█▋        | 6/36 [00:01<00:09,  3.19it/s]
⏎ Capturing batches (bs=208 avail_mem=10.62 GB):  19%|█▉        | 7/36 [00:02<00:08,  3.22it/s]
⏎ Capturing batches (bs=200 avail_mem=10.62 GB):  19%|█▉        | 7/36 [00:02<00:08,  3.22it/s]
⏎ Capturing batches (bs=200 avail_mem=10.62 GB):  22%|██▏       | 8/36 [00:02<00:08,  3.25it/s]
⏎ Capturing batches (bs=192 avail_mem=10.62 GB):  22%|██▏       | 8/36 [00:02<00:08,  3.25it/s]
⏎ Capturing batches (bs=192 avail_mem=10.62 GB):  25%|██▌       | 9/36 [00:02<00:08,  3.26it/s]
⏎ Capturing batches (bs=184 avail_mem=10.61 GB):  25%|██▌       | 9/36 [00:02<00:08,  3.26it/s]
⏎ Capturing batches (bs=184 avail_mem=10.61 GB):  28%|██▊       | 10/36 [00:03<00:07,  3.28it/s]
⏎ Capturing batches (bs=176 avail_mem=10.61 GB):  28%|██▊       | 10/36 [00:03<00:07,  3.28it/s]
⏎ Capturing batches (bs=176 avail_mem=10.61 GB):  31%|███       | 11/36 [00:03<00:07,  3.26it/s]
⏎ Capturing batches (bs=168 avail_mem=10.61 GB):  31%|███       | 11/36 [00:03<00:07,  3.26it/s]
⏎ Capturing batches (bs=168 avail_mem=10.61 GB):  33%|███▎      | 12/36 [00:03<00:07,  3.24it/s]
⏎ Capturing batches (bs=160 avail_mem=10.61 GB):  33%|███▎      | 12/36 [00:03<00:07,  3.24it/s]
⏎ Capturing batches (bs=160 avail_mem=10.61 GB):  36%|███▌      | 13/36 [00:04<00:07,  3.21it/s]
⏎ Capturing batches (bs=152 avail_mem=10.60 GB):  36%|███▌      | 13/36 [00:04<00:07,  3.21it/s]
⏎ Capturing batches (bs=152 avail_mem=10.60 GB):  39%|███▉      | 14/36 [00:04<00:06,  3.23it/s]
⏎ Capturing batches (bs=144 avail_mem=10.60 GB):  39%|███▉      | 14/36 [00:04<00:06,  3.23it/s]
⏎ Capturing batches (bs=144 avail_mem=10.60 GB):  42%|████▏     | 15/36 [00:04<00:06,  3.24it/s]
⏎ Capturing batches (bs=136 avail_mem=10.60 GB):  42%|████▏     | 15/36 [00:04<00:06,  3.24it/s]
⏎ Capturing batches (bs=136 avail_mem=10.60 GB):  44%|████▍     | 16/36 [00:04<00:06,  3.27it/s]
⏎ Capturing batches (bs=128 avail_mem=10.60 GB):  44%|████▍     | 16/36 [00:04<00:06,  3.27it/s]
⏎ Capturing batches (bs=128 avail_mem=10.60 GB):  47%|████▋     | 17/36 [00:05<00:05,  3.29it/s]
⏎ Capturing batches (bs=120 avail_mem=10.60 GB):  47%|████▋     | 17/36 [00:05<00:05,  3.29it/s]
⏎ Capturing batches (bs=120 avail_mem=10.60 GB):  50%|█████     | 18/36 [00:05<00:05,  3.26it/s]
⏎ Capturing batches (bs=112 avail_mem=10.59 GB):  50%|█████     | 18/36 [00:05<00:05,  3.26it/s]
⏎ Capturing batches (bs=112 avail_mem=10.59 GB):  53%|█████▎    | 19/36 [00:05<00:05,  3.26it/s]
⏎ Capturing batches (bs=104 avail_mem=10.59 GB):  53%|█████▎    | 19/36 [00:05<00:05,  3.26it/s]
⏎ Capturing batches (bs=104 avail_mem=10.59 GB):  56%|█████▌    | 20/36 [00:06<00:04,  3.23it/s]
⏎ Capturing batches (bs=96 avail_mem=10.59 GB):  56%|█████▌    | 20/36 [00:06<00:04,  3.23it/s] 
⏎ Capturing batches (bs=96 avail_mem=10.59 GB):  58%|█████▊    | 21/36 [00:06<00:04,  3.25it/s]
⏎ Capturing batches (bs=88 avail_mem=10.58 GB):  58%|█████▊    | 21/36 [00:06<00:04,  3.25it/s]
⏎ Capturing batches (bs=88 avail_mem=10.58 GB):  61%|██████    | 22/36 [00:06<00:04,  3.24it/s]
⏎ Capturing batches (bs=80 avail_mem=10.58 GB):  61%|██████    | 22/36 [00:06<00:04,  3.24it/s]
⏎ Capturing batches (bs=80 avail_mem=10.58 GB):  64%|██████▍   | 23/36 [00:07<00:04,  3.24it/s]
⏎ Capturing batches (bs=72 avail_mem=10.58 GB):  64%|██████▍   | 23/36 [00:07<00:04,  3.24it/s]
⏎ Capturing batches (bs=72 avail_mem=10.58 GB):  67%|██████▋   | 24/36 [00:07<00:03,  3.23it/s]
⏎ Capturing batches (bs=64 avail_mem=10.57 GB):  67%|██████▋   | 24/36 [00:07<00:03,  3.23it/s]
⏎ Capturing batches (bs=64 avail_mem=10.57 GB):  69%|██████▉   | 25/36 [00:07<00:03,  2.98it/s]
⏎ Capturing batches (bs=56 avail_mem=10.57 GB):  69%|██████▉   | 25/36 [00:07<00:03,  2.98it/s]
⏎ Capturing batches (bs=56 avail_mem=10.57 GB):  72%|███████▏  | 26/36 [00:08<00:03,  3.04it/s]
⏎ Capturing batches (bs=48 avail_mem=10.57 GB):  72%|███████▏  | 26/36 [00:08<00:03,  3.04it/s]
⏎ Capturing batches (bs=48 avail_mem=10.57 GB):  75%|███████▌  | 27/36 [00:08<00:02,  3.10it/s]
⏎ Capturing batches (bs=40 avail_mem=10.57 GB):  75%|███████▌  | 27/36 [00:08<00:02,  3.10it/s]
⏎ Capturing batches (bs=40 avail_mem=10.57 GB):  78%|███████▊  | 28/36 [00:08<00:02,  3.13it/s]
⏎ Capturing batches (bs=32 avail_mem=10.56 GB):  78%|███████▊  | 28/36 [00:08<00:02,  3.13it/s]
⏎ Capturing batches (bs=32 avail_mem=10.56 GB):  81%|████████  | 29/36 [00:09<00:02,  3.12it/s]
⏎ Capturing batches (bs=24 avail_mem=10.56 GB):  81%|████████  | 29/36 [00:09<00:02,  3.12it/s]
⏎ Capturing batches (bs=24 avail_mem=10.56 GB):  83%|████████▎ | 30/36 [00:09<00:01,  3.09it/s]
⏎ Capturing batches (bs=16 avail_mem=10.56 GB):  83%|████████▎ | 30/36 [00:09<00:01,  3.09it/s]
⏎ Capturing batches (bs=16 avail_mem=10.56 GB):  86%|████████▌ | 31/36 [00:09<00:01,  3.07it/s]
⏎ Capturing batches (bs=12 avail_mem=10.56 GB):  86%|████████▌ | 31/36 [00:09<00:01,  3.07it/s]
⏎ Capturing batches (bs=12 avail_mem=10.56 GB):  89%|████████▉ | 32/36 [00:10<00:01,  3.06it/s]
⏎ Capturing batches (bs=8 avail_mem=10.55 GB):  89%|████████▉ | 32/36 [00:10<00:01,  3.06it/s] 
⏎ Capturing batches (bs=8 avail_mem=10.55 GB):  92%|█████████▏| 33/36 [00:10<00:00,  3.03it/s]
⏎ Capturing batches (bs=4 avail_mem=10.55 GB):  92%|█████████▏| 33/36 [00:10<00:00,  3.03it/s]
⏎ Capturing batches (bs=4 avail_mem=10.55 GB):  94%|█████████▍| 34/36 [00:10<00:00,  3.04it/s]
⏎ Capturing batches (bs=2 avail_mem=10.55 GB):  94%|█████████▍| 34/36 [00:10<00:00,  3.04it/s]
⏎ Capturing batches (bs=2 avail_mem=10.55 GB):  97%|█████████▋| 35/36 [00:11<00:00,  3.03it/s]
⏎ Capturing batches (bs=1 avail_mem=10.55 GB):  97%|█████████▋| 35/36 [00:11<00:00,  3.03it/s]
⏎ Capturing batches (bs=1 avail_mem=10.55 GB): 100%|██████████| 36/36 [00:11<00:00,  2.86it/s]
⏎ Capturing batches (bs=1 avail_mem=10.55 GB): 100%|██████████| 36/36 [00:11<00:00,  3.14it/s]
[2026-08-27 03:46:30] Capture target decode CUDA graph end. elapsed=12.26 s, mem usage=0.34 GB, avail mem=10.54 GB.
[2026-08-27 03:46:30] max_total_num_tokens=392920, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=40960, available_gpu_mem=10.54 GB
[2026-08-27 03:46:30] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 03:46:30] Engine startup timings (s): load_weight=12.02, kv_cache_allocation=0.02, scheduler_e2e=52.09, cuda_graph={prefill=22.96, decode=12.26, target_verify=0.00, draft_prefill=0.00, draft_decode=0.00, draft_extend=0.00}, tokenizer_e2e=65.82
[2026-08-27 03:46:30] INFO:     Started server process [6748]
[2026-08-27 03:46:30] INFO:     Waiting for application startup.
[2026-08-27 03:46:30] Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}
[2026-08-27 03:46:30] INFO:     Application startup complete.
[2026-08-27 03:46:30] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2026-08-27 03:46:31] INFO:     127.0.0.1:50386 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 03:46:31] INFO:     127.0.0.1:50388 - "GET /model_info HTTP/1.1" 200 OK
[2026-08-27 03:46:31] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 3.50
[2026-08-27 03:46:32] INFO:     127.0.0.1:50390 - "POST /generate HTTP/1.1" 200 OK
[2026-08-27 03:46:32] Freezing GC in Scheduler process. gen0: 48->0, gen1: 2763->0, gen2: 1060294->0
[2026-08-27 03:46:32] Freezing GC in Tokenizer Manager process. gen0: 366->0, gen1: 4764->0, gen2: 911861->0
[2026-08-27 03:46:32] INFO:     127.0.0.1:40016 - "POST /freeze_gc HTTP/1.1" 200 OK
[2026-08-27 03:46:32] The server is fired up and ready to roll!
[2026-08-27 03:46:32] Freezing GC in Detokenizer Manager process. gen0: 382->0, gen1: 0->0, gen2: 851112->0
[2026-08-27 03:46:36] Prefill batch, #new-seq: 1, #new-token: 1, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.22
[2026-08-27 03:46:37] INFO:     127.0.0.1:40022 - "GET /health HTTP/1.1" 200 OK
[2026-08-27 03:46:37] Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 2, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 3.26
[2026-08-27 03:46:37] Decode batch, #running-req: 1, #token: 0, token usage: 0.00, cuda graph: True, gen throughput (token/s): 5.41, #queue-req: 0
[2026-08-27 03:46:37] INFO:     127.0.0.1:40038 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:52] INFO:     127.0.0.1:54804 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 03:46:52] INFO:     127.0.0.1:54804 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 03:46:54] Triton kernel '_fwd_kernel' took 1.08 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 03:46:54] Prefill batch, #new-seq: 1, #new-token: 1018, #cached-token: 0, token usage: 0.02, #running-req: 0, #queue-req: 6, #pending-token: 0, cuda graph: True, input throughput (token/s): 60.36
[2026-08-27 03:46:54] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] Prefill batch, #new-seq: 9, #new-token: 8192, #cached-token: 0, token usage: 0.03, #running-req: 1, #queue-req: 0, #pending-token: 6993, cuda graph: True, input throughput (token/s): 413282.46
[2026-08-27 03:46:54] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] Prefill batch, #new-seq: 7, #new-token: 5787, #cached-token: 1206, token usage: 0.03, #running-req: 9, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 67494.66
[2026-08-27 03:46:54] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:54] Decode batch, #running-req: 16, #token: 13814, token usage: 0.04, cuda graph: True, gen throughput (token/s): 36.50, #queue-req: 0
[2026-08-27 03:46:55] Decode batch, #running-req: 16, #token: 14454, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1686.52, #queue-req: 0
[2026-08-27 03:46:55] Decode batch, #running-req: 16, #token: 15094, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1657.36, #queue-req: 0
[2026-08-27 03:46:55] Decode batch, #running-req: 16, #token: 15734, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1657.85, #queue-req: 0
[2026-08-27 03:46:56] Decode batch, #running-req: 16, #token: 16374, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1645.14, #queue-req: 0
[2026-08-27 03:46:56] Decode batch, #running-req: 16, #token: 17014, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1642.19, #queue-req: 0
[2026-08-27 03:46:56] Prefill batch, #new-seq: 1, #new-token: 823, #cached-token: 201, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 336.16
[2026-08-27 03:46:56] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:56] Prefill batch, #new-seq: 4, #new-token: 3241, #cached-token: 826, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 62256.94
[2026-08-27 03:46:56] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:56] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:56] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:56] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] Prefill batch, #new-seq: 6, #new-token: 4821, #cached-token: 1208, token usage: 0.03, #running-req: 5, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 56633.75
[2026-08-27 03:46:57] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] Prefill batch, #new-seq: 5, #new-token: 4082, #cached-token: 1005, token usage: 0.03, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 58929.05
[2026-08-27 03:46:57] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:57] Decode batch, #running-req: 16, #token: 13575, token usage: 0.03, cuda graph: True, gen throughput (token/s): 1084.32, #queue-req: 0
[2026-08-27 03:46:57] Decode batch, #running-req: 16, #token: 14215, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1765.17, #queue-req: 0
[2026-08-27 03:46:58] Decode batch, #running-req: 16, #token: 14855, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1711.39, #queue-req: 0
[2026-08-27 03:46:58] Decode batch, #running-req: 16, #token: 15495, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1682.68, #queue-req: 0
[2026-08-27 03:46:58] Decode batch, #running-req: 16, #token: 16135, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1663.95, #queue-req: 0
[2026-08-27 03:46:59] Decode batch, #running-req: 16, #token: 16775, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1657.96, #queue-req: 0
[2026-08-27 03:46:59] Prefill batch, #new-seq: 1, #new-token: 815, #cached-token: 202, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 336.63
[2026-08-27 03:46:59] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] Prefill batch, #new-seq: 3, #new-token: 2442, #cached-token: 604, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 56305.78
[2026-08-27 03:46:59] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] Prefill batch, #new-seq: 6, #new-token: 4844, #cached-token: 1208, token usage: 0.03, #running-req: 4, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 56573.68
[2026-08-27 03:46:59] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] Prefill batch, #new-seq: 6, #new-token: 4913, #cached-token: 1207, token usage: 0.03, #running-req: 10, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 57704.68
[2026-08-27 03:46:59] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:46:59] Decode batch, #running-req: 16, #token: 13345, token usage: 0.03, cuda graph: True, gen throughput (token/s): 1039.07, #queue-req: 0
[2026-08-27 03:47:00] Decode batch, #running-req: 16, #token: 13985, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1840.26, #queue-req: 0
[2026-08-27 03:47:00] Decode batch, #running-req: 16, #token: 14625, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1685.94, #queue-req: 0
[2026-08-27 03:47:00] Decode batch, #running-req: 16, #token: 15265, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1681.92, #queue-req: 0
[2026-08-27 03:47:01] Decode batch, #running-req: 16, #token: 15905, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1677.62, #queue-req: 0
[2026-08-27 03:47:01] Decode batch, #running-req: 16, #token: 16545, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1663.50, #queue-req: 0
[2026-08-27 03:47:02] Decode batch, #running-req: 16, #token: 17185, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1644.29, #queue-req: 0
[2026-08-27 03:47:02] Prefill batch, #new-seq: 1, #new-token: 818, #cached-token: 201, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 336.25
[2026-08-27 03:47:02] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] Prefill batch, #new-seq: 3, #new-token: 2448, #cached-token: 605, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 54618.91
[2026-08-27 03:47:02] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] Prefill batch, #new-seq: 5, #new-token: 4095, #cached-token: 1006, token usage: 0.03, #running-req: 4, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 58569.47
[2026-08-27 03:47:02] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] Prefill batch, #new-seq: 7, #new-token: 5705, #cached-token: 1421, token usage: 0.03, #running-req: 9, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 57530.73
[2026-08-27 03:47:02] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:02] Decode batch, #running-req: 16, #token: 13795, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1105.69, #queue-req: 0
[2026-08-27 03:47:03] Decode batch, #running-req: 16, #token: 14435, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1748.67, #queue-req: 0
[2026-08-27 03:47:03] Decode batch, #running-req: 16, #token: 15075, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1688.94, #queue-req: 0
[2026-08-27 03:47:03] Decode batch, #running-req: 16, #token: 15715, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1653.61, #queue-req: 0
[2026-08-27 03:47:04] Decode batch, #running-req: 16, #token: 16355, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1672.29, #queue-req: 0
[2026-08-27 03:47:04] Decode batch, #running-req: 16, #token: 16995, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1650.82, #queue-req: 0
[2026-08-27 03:47:04] Prefill batch, #new-seq: 1, #new-token: 789, #cached-token: 201, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 325.90
[2026-08-27 03:47:04] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] Prefill batch, #new-seq: 3, #new-token: 2428, #cached-token: 615, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 56021.53
[2026-08-27 03:47:04] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] Prefill batch, #new-seq: 6, #new-token: 4861, #cached-token: 1220, token usage: 0.03, #running-req: 4, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 56856.01
[2026-08-27 03:47:04] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:04] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:05] Prefill batch, #new-seq: 6, #new-token: 4869, #cached-token: 1210, token usage: 0.03, #running-req: 10, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 57774.02
[2026-08-27 03:47:05] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:05] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:05] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:05] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:05] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:05] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:05] Decode batch, #running-req: 16, #token: 13421, token usage: 0.03, cuda graph: True, gen throughput (token/s): 1070.47, #queue-req: 0
[2026-08-27 03:47:05] Decode batch, #running-req: 16, #token: 14061, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1821.56, #queue-req: 0
[2026-08-27 03:47:05] Decode batch, #running-req: 16, #token: 14701, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1675.82, #queue-req: 0
[2026-08-27 03:47:06] Decode batch, #running-req: 16, #token: 15341, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1671.50, #queue-req: 0
[2026-08-27 03:47:06] Decode batch, #running-req: 16, #token: 15981, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1669.36, #queue-req: 0
[2026-08-27 03:47:07] Decode batch, #running-req: 16, #token: 16621, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1646.63, #queue-req: 0
[2026-08-27 03:47:07] Decode batch, #running-req: 16, #token: 0, token usage: 0.00, cuda graph: True, gen throughput (token/s): 1632.61, #queue-req: 0
[2026-08-27 03:47:07] Prefill batch, #new-seq: 1, #new-token: 822, #cached-token: 202, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 338.15
[2026-08-27 03:47:07] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] Prefill batch, #new-seq: 3, #new-token: 2407, #cached-token: 605, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 52944.73
[2026-08-27 03:47:07] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] Prefill batch, #new-seq: 5, #new-token: 4056, #cached-token: 1008, token usage: 0.03, #running-req: 4, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 58198.85
[2026-08-27 03:47:07] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] Prefill batch, #new-seq: 7, #new-token: 5660, #cached-token: 1422, token usage: 0.03, #running-req: 9, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 56993.41
[2026-08-27 03:47:07] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:07] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:08] Decode batch, #running-req: 16, #token: 13803, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1116.78, #queue-req: 0
[2026-08-27 03:47:08] Decode batch, #running-req: 16, #token: 14443, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1715.61, #queue-req: 0
[2026-08-27 03:47:08] Decode batch, #running-req: 16, #token: 15083, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1671.69, #queue-req: 0
[2026-08-27 03:47:09] Decode batch, #running-req: 16, #token: 15723, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1674.30, #queue-req: 0
[2026-08-27 03:47:09] Decode batch, #running-req: 16, #token: 16363, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1676.57, #queue-req: 0
[2026-08-27 03:47:09] Decode batch, #running-req: 16, #token: 17003, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1648.42, #queue-req: 0
[2026-08-27 03:47:10] Prefill batch, #new-seq: 1, #new-token: 809, #cached-token: 202, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 333.97
[2026-08-27 03:47:10] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] Prefill batch, #new-seq: 4, #new-token: 3262, #cached-token: 807, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 59610.28
[2026-08-27 03:47:10] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] Prefill batch, #new-seq: 5, #new-token: 4083, #cached-token: 1008, token usage: 0.03, #running-req: 5, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 58635.68
[2026-08-27 03:47:10] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] Prefill batch, #new-seq: 6, #new-token: 4837, #cached-token: 1241, token usage: 0.03, #running-req: 10, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 56821.92
[2026-08-27 03:47:10] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:10] Decode batch, #running-req: 16, #token: 13607, token usage: 0.03, cuda graph: True, gen throughput (token/s): 1113.29, #queue-req: 0
[2026-08-27 03:47:10] Decode batch, #running-req: 16, #token: 14247, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1779.79, #queue-req: 0
[2026-08-27 03:47:11] Decode batch, #running-req: 16, #token: 14887, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1683.21, #queue-req: 0
[2026-08-27 03:47:11] Decode batch, #running-req: 16, #token: 15527, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1660.36, #queue-req: 0
[2026-08-27 03:47:12] Decode batch, #running-req: 16, #token: 16167, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1652.85, #queue-req: 0
[2026-08-27 03:47:12] Decode batch, #running-req: 16, #token: 16807, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1658.74, #queue-req: 0
[2026-08-27 03:47:12] Prefill batch, #new-seq: 1, #new-token: 809, #cached-token: 209, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 334.13
[2026-08-27 03:47:12] INFO:     127.0.0.1:54804 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] Prefill batch, #new-seq: 4, #new-token: 3266, #cached-token: 810, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 62193.74
[2026-08-27 03:47:12] INFO:     127.0.0.1:54832 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54834 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54896 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54910 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] Prefill batch, #new-seq: 6, #new-token: 4871, #cached-token: 1210, token usage: 0.03, #running-req: 5, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 57431.37
[2026-08-27 03:47:12] INFO:     127.0.0.1:54932 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54936 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54898 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54836 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54882 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] Prefill batch, #new-seq: 5, #new-token: 3988, #cached-token: 1042, token usage: 0.03, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 57445.92
[2026-08-27 03:47:12] INFO:     127.0.0.1:54852 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54912 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54820 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54916 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:12] INFO:     127.0.0.1:54866 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:47:13] Decode batch, #running-req: 16, #token: 13317, token usage: 0.03, cuda graph: True, gen throughput (token/s): 1052.15, #queue-req: 0
[2026-08-27 03:47:13] Decode batch, #running-req: 16, #token: 13957, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1881.06, #queue-req: 0
[2026-08-27 03:47:13] Decode batch, #running-req: 16, #token: 14597, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1700.55, #queue-req: 0
[2026-08-27 03:47:14] Decode batch, #running-req: 16, #token: 15237, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1682.98, #queue-req: 0
[2026-08-27 03:47:14] Decode batch, #running-req: 16, #token: 15877, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1670.87, #queue-req: 0
[2026-08-27 03:47:14] Decode batch, #running-req: 16, #token: 16517, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1659.61, #queue-req: 0
[2026-08-27 03:47:15] Decode batch, #running-req: 16, #token: 17157, token usage: 0.04, cuda graph: True, gen throughput (token/s): 1674.60, #queue-req: 0
[2026-08-27 03:47:15] INFO:     127.0.0.1:54820 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 03:47:15] INFO:     127.0.0.1:54804 - "GET /metrics HTTP/1.1" 404 Not Found
```

## 11. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/smoke.json`

- 바이트 **509** · 줄 **1** · SHA256 `803bd5d641fed2958ab5168d98c61489980331ce8ccf9e3bd720d823c933b6c0`
- 인코딩 utf-8

정리한 형태:

```json
{
  "id": "d5b91e88898741ce8df85f0132cf9406",
  "object": "text_completion",
  "created": 1787802397,
  "model": "qwen30b",
  "choices": [
    {
      "index": 0,
      "text": " Paris. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid.",
      "logprobs": null,
      "finish_reason": "length",
      "matched_stop": null
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 37,
    "completion_tokens": 32,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

원문:

```text
{"id":"d5b91e88898741ce8df85f0132cf9406","object":"text_completion","created":1787802397,"model":"qwen30b","choices":[{"index":0,"text":" Paris. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid.","logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":5,"total_tokens":37,"completion_tokens":32,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
```

## 12. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/.cpu_mon`

- 바이트 **8** · 줄 **1** · SHA256 `12d33260a84c5c3febf254bf3d7470dac08c1366a1396990400284f0d7fffa1d`
- 인코딩 utf-8

```text
4146516
```

## 13. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/.gpu_mon`

- 바이트 **8** · 줄 **1** · SHA256 `2e85fbff98393e5684155bcb49ceee2b40b61c57ab8285a35e66d638d93bc17f`
- 인코딩 utf-8

```text
4146514
```

## 14. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/bench.log`

- 바이트 **5,312** · 줄 **48** · SHA256 `80a2254d10f80835bb6fd8e7a043614e730ea35482a5fbc101ac2f6c786c6387`
- 인코딩 utf-8 · CR 10건 치환

```text
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f95c1fbede0>, trust_remote_code=False, seed=42, num_prompts=128, dataset_name='sonnet', no_stream=False, dataset_path='/repo/benchmarks/sonnet.txt', no_oversample=False, skip_chat_template=False, enable_multimodal_chat=False, disable_shuffle=False, custom_output_len=256, custom_ensure_client_side_data=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=1024, sonnet_output_len=256, sonnet_prefix_len=200, sharegpt_output_len=None, timed_trace_chunk_hash_size=16, timed_trace_sec_multiplier=1, timed_trace_label_timestamp='timestamp', timed_trace_label_input_length='input_length', timed_trace_label_output_length='output_length', timed_trace_label_hash_ids='hash_ids', blazedit_min_distance=0.0, blazedit_max_distance=1.0, asr_max_audio_len_sec=inf, asr_min_audio_len_sec=0.0, random_input_len=1024, random_output_len=128, random_range_ratio='0.0', random_prefix_len=0, random_batch_size=1, no_reranker=False, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 1}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, bfcl_categories=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, speed_bench_dataset_subset='qualitative', speed_bench_output_len=4096, speed_bench_category=None, label=None, backend='openai', base_url='http://127.0.0.1:30000', host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=16, model='qwen30b', input_len=None, output_len=None, tokenizer='/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782', tokenizer_mode='auto', use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, probe_request_rate=0.0, disable_tqdm=False, num_warmups=0, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, self_timed=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='50,95', goodput=None, request_id_prefix='bench-a5f108a9-', top_p=None, top_k=None, min_p=None, temperature=None, frequency_penalty=None, presence_penalty=None, repetition_penalty=None, served_model_name=None, lora_modules=None, lora_assignment='random', ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=0, chat_template_kwargs=None, extra_body=None, skip_tokenizer_init=False, insecure=False, plot_timeline=False, timeline_itl_thresholds='25,50', plot_dataset_stats=False)
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16

⏎   0%|          | 0/128 [00:00<?, ?it/s]
⏎   1%|          | 1/128 [00:43<1:32:06, 43.52s/it]
⏎  13%|█▎        | 17/128 [01:26<08:05,  4.38s/it] 
⏎  26%|██▌       | 33/128 [02:08<05:21,  3.39s/it]
⏎  38%|███▊      | 49/128 [02:51<04:01,  3.06s/it]
⏎  51%|█████     | 65/128 [03:34<03:03,  2.91s/it]
⏎  63%|██████▎   | 81/128 [04:16<02:12,  2.82s/it]
⏎  76%|███████▌  | 97/128 [05:00<01:26,  2.79s/it]
⏎  88%|████████▊ | 113/128 [05:41<00:40,  2.71s/it]
⏎ 100%|██████████| 128/128 [05:41<00:00,  2.66s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  341.03    
Total input tokens:                      129773    
Total generated tokens:                  32768     
Request throughput (req/s):              0.38      
Output token throughput (tok/s):         96.09     
Peak output token throughput (tok/s):    128.00    
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          476.62    
---------------Time to First Token----------------
Mean TTFT (ms):                          2596.16   
Median TTFT (ms):                        2381.02   
P50 TTFT (ms):                           2381.02   
P95 TTFT (ms):                           3719.52   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          156.98    
Median TPOT (ms):                        157.69    
P50 TPOT (ms):                           157.69    
P95 TPOT (ms):                           162.29    
---------------Inter-token Latency----------------
Mean ITL (ms):                           156.98    
Median ITL (ms):                         154.83    
P50 ITL (ms):                            154.83    
P95 ITL (ms):                            177.77    
==================================================
```

## 15. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/cpu_util.txt`

- 바이트 **3,931** · 줄 **179** · SHA256 `0985cc34398398ab30f9677563b5e5fd4ae2db4c364ea51bc16d1bdb00e317ce`
- 인코딩 utf-8

```text
1787802765 busy=42.40
1787802767 busy=44.27
1787802769 busy=22.56
1787802771 busy=2.06
1787802773 busy=1.94
1787802775 busy=1.83
1787802777 busy=2.09
1787802779 busy=5.71
1787802781 busy=1.82
1787802783 busy=2.25
1787802785 busy=42.79
1787802787 busy=44.36
1787802789 busy=44.90
1787802791 busy=44.58
1787802793 busy=44.67
1787802795 busy=44.86
1787802797 busy=45.14
1787802799 busy=44.86
1787802801 busy=44.67
1787802803 busy=44.82
1787802805 busy=44.89
1787802807 busy=45.76
1787802809 busy=44.99
1787802811 busy=44.68
1787802813 busy=45.13
1787802815 busy=45.20
1787802817 busy=44.58
1787802819 busy=44.86
1787802821 busy=44.84
1787802823 busy=45.68
1787802825 busy=44.67
1787802827 busy=44.99
1787802829 busy=45.37
1787802831 busy=44.71
1787802833 busy=44.93
1787802835 busy=44.45
1787802837 busy=44.76
1787802839 busy=44.79
1787802841 busy=44.70
1787802843 busy=45.26
1787802845 busy=44.70
1787802847 busy=44.74
1787802849 busy=44.79
1787802851 busy=44.67
1787802853 busy=44.89
1787802855 busy=44.66
1787802857 busy=44.60
1787802859 busy=44.74
1787802861 busy=44.66
1787802863 busy=44.95
1787802865 busy=44.79
1787802867 busy=44.85
1787802869 busy=44.92
1787802871 busy=45.60
1787802873 busy=45.26
1787802875 busy=45.31
1787802877 busy=44.68
1787802879 busy=44.62
1787802881 busy=44.68
1787802883 busy=44.70
1787802885 busy=44.96
1787802887 busy=44.83
1787802889 busy=44.66
1787802891 busy=44.75
1787802893 busy=44.77
1787802895 busy=44.55
1787802897 busy=44.96
1787802899 busy=44.57
1787802901 busy=44.77
1787802903 busy=45.22
1787802905 busy=44.78
1787802907 busy=44.84
1787802909 busy=45.38
1787802911 busy=44.67
1787802913 busy=45.31
1787802915 busy=45.50
1787802917 busy=44.80
1787802919 busy=45.15
1787802921 busy=44.58
1787802923 busy=44.79
1787802925 busy=44.83
1787802927 busy=45.78
1787802929 busy=44.77
1787802932 busy=44.66
1787802934 busy=45.05
1787802936 busy=45.20
1787802938 busy=44.89
1787802940 busy=44.54
1787802942 busy=44.71
1787802944 busy=44.77
1787802946 busy=44.70
1787802948 busy=45.35
1787802950 busy=44.73
1787802952 busy=44.65
1787802954 busy=45.04
1787802956 busy=45.03
1787802958 busy=45.30
1787802960 busy=44.63
1787802962 busy=44.47
1787802964 busy=45.31
1787802966 busy=44.69
1787802968 busy=44.74
1787802970 busy=44.76
1787802972 busy=44.81
1787802974 busy=44.71
1787802976 busy=44.61
1787802978 busy=44.63
1787802980 busy=44.71
1787802982 busy=44.83
1787802984 busy=45.02
1787802986 busy=44.80
1787802988 busy=47.74
1787802990 busy=44.93
1787802992 busy=44.69
1787802994 busy=45.17
1787802996 busy=45.20
1787802998 busy=45.02
1787803000 busy=45.51
1787803002 busy=44.62
1787803004 busy=44.63
1787803006 busy=44.58
1787803008 busy=44.55
1787803010 busy=45.97
1787803012 busy=44.79
1787803014 busy=44.96
1787803016 busy=44.57
1787803018 busy=44.61
1787803020 busy=44.85
1787803022 busy=44.52
1787803024 busy=45.28
1787803026 busy=44.55
1787803028 busy=44.87
1787803030 busy=44.61
1787803032 busy=44.53
1787803034 busy=44.72
1787803036 busy=45.15
1787803038 busy=45.22
1787803040 busy=44.93
1787803042 busy=45.43
1787803044 busy=44.98
1787803046 busy=44.40
1787803048 busy=44.85
1787803050 busy=44.52
1787803052 busy=44.55
1787803054 busy=45.36
1787803056 busy=44.85
1787803058 busy=44.56
1787803060 busy=44.59
1787803062 busy=44.54
1787803064 busy=44.68
1787803066 busy=44.44
1787803068 busy=44.44
1787803070 busy=44.90
1787803072 busy=44.49
1787803074 busy=44.98
1787803076 busy=44.71
1787803078 busy=44.55
1787803080 busy=44.54
1787803082 busy=44.44
1787803084 busy=45.43
1787803086 busy=45.08
1787803088 busy=44.80
1787803090 busy=44.39
1787803092 busy=44.80
1787803094 busy=44.58
1787803096 busy=44.58
1787803098 busy=44.49
1787803100 busy=44.50
1787803102 busy=45.18
1787803104 busy=45.08
1787803106 busy=45.57
1787803108 busy=44.83
1787803110 busy=44.81
1787803112 busy=44.69
1787803114 busy=45.64
1787803116 busy=44.89
1787803118 busy=44.71
1787803120 busy=44.55
1787803122 busy=44.81
```

## 16. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/gpu_util.csv`

- 바이트 **13,798** · 줄 **584** · SHA256 `6a734d67e10d12031cb08cfd152145a38d7cd55b90473cfa38a7da5ae8f549bc`
- 인코딩 utf-8

```text
0, 0 %, 69246 MiB, 116.56 W
1, 0 %, 4 MiB, 68.60 W
2, 0 %, 4 MiB, 69.10 W
3, 0 %, 4 MiB, 69.31 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.76 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.82 W
0, 0 %, 69246 MiB, 117.22 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.05 W
3, 0 %, 4 MiB, 69.28 W
4, 0 %, 4 MiB, 69.67 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.83 W
0, 0 %, 69246 MiB, 115.69 W
1, 0 %, 4 MiB, 68.62 W
2, 0 %, 4 MiB, 69.15 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.68 W
5, 0 %, 4 MiB, 68.83 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.86 W
0, 0 %, 69246 MiB, 115.75 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.19 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.68 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.85 W
0, 0 %, 69324 MiB, 115.83 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.15 W
3, 0 %, 4 MiB, 69.29 W
4, 0 %, 4 MiB, 69.70 W
5, 0 %, 4 MiB, 68.74 W
6, 0 %, 4 MiB, 70.15 W
7, 0 %, 4 MiB, 70.82 W
0, 5 %, 70064 MiB, 119.97 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.10 W
3, 0 %, 4 MiB, 69.30 W
4, 0 %, 4 MiB, 69.68 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.87 W
0, 5 %, 70064 MiB, 119.76 W
1, 0 %, 4 MiB, 68.72 W
2, 0 %, 4 MiB, 69.12 W
3, 0 %, 4 MiB, 69.35 W
4, 0 %, 4 MiB, 69.65 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.83 W
0, 5 %, 70064 MiB, 120.62 W
1, 0 %, 4 MiB, 68.71 W
2, 0 %, 4 MiB, 69.12 W
3, 0 %, 4 MiB, 69.29 W
4, 0 %, 4 MiB, 69.70 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.83 W
0, 5 %, 70064 MiB, 120.29 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.17 W
3, 0 %, 4 MiB, 69.27 W
4, 0 %, 4 MiB, 69.70 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.12 W
7, 0 %, 4 MiB, 70.81 W
0, 5 %, 70064 MiB, 121.49 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.64 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.13 W
7, 0 %, 4 MiB, 70.85 W
0, 5 %, 70064 MiB, 120.43 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.13 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.66 W
5, 0 %, 4 MiB, 68.72 W
6, 0 %, 4 MiB, 70.22 W
7, 0 %, 4 MiB, 70.86 W
0, 5 %, 70064 MiB, 120.29 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.10 W
3, 0 %, 4 MiB, 69.31 W
4, 0 %, 4 MiB, 69.73 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.25 W
7, 0 %, 4 MiB, 70.82 W
0, 5 %, 70064 MiB, 120.68 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.09 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.65 W
5, 0 %, 4 MiB, 68.81 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.85 W
0, 10 %, 70320 MiB, 142.66 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.13 W
3, 0 %, 4 MiB, 69.34 W
4, 0 %, 4 MiB, 69.68 W
5, 0 %, 4 MiB, 68.77 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.84 W
0, 5 %, 70320 MiB, 120.75 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.16 W
3, 0 %, 4 MiB, 69.31 W
4, 0 %, 4 MiB, 69.75 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.11 W
7, 0 %, 4 MiB, 70.85 W
0, 5 %, 70320 MiB, 120.60 W
1, 0 %, 4 MiB, 68.71 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.69 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.16 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 121.53 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.08 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.65 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.86 W
0, 5 %, 70320 MiB, 120.93 W
1, 0 %, 4 MiB, 68.73 W
2, 0 %, 4 MiB, 69.08 W
3, 0 %, 4 MiB, 69.29 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.74 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.83 W
0, 5 %, 70320 MiB, 121.10 W
1, 0 %, 4 MiB, 68.69 W
2, 0 %, 4 MiB, 69.10 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.75 W
5, 0 %, 4 MiB, 68.81 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 120.72 W
1, 0 %, 4 MiB, 68.69 W
2, 0 %, 4 MiB, 69.19 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.87 W
0, 5 %, 70320 MiB, 121.27 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.21 W
3, 0 %, 4 MiB, 69.30 W
4, 0 %, 4 MiB, 69.70 W
5, 0 %, 4 MiB, 68.81 W
6, 0 %, 4 MiB, 70.14 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 121.07 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.17 W
3, 0 %, 4 MiB, 69.33 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.74 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.85 W
0, 5 %, 70320 MiB, 122.82 W
1, 0 %, 4 MiB, 68.63 W
2, 0 %, 4 MiB, 69.12 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 121.19 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.34 W
4, 0 %, 4 MiB, 69.74 W
5, 0 %, 4 MiB, 68.82 W
6, 0 %, 4 MiB, 70.23 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 121.56 W
1, 0 %, 4 MiB, 68.73 W
2, 0 %, 4 MiB, 69.17 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.69 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.16 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 120.83 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.19 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.78 W
5, 0 %, 4 MiB, 68.82 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.90 W
0, 5 %, 70320 MiB, 120.56 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.15 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.11 W
7, 0 %, 4 MiB, 70.89 W
0, 5 %, 70320 MiB, 120.73 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.09 W
3, 0 %, 4 MiB, 69.31 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.92 W
0, 5 %, 70320 MiB, 121.49 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.06 W
3, 0 %, 4 MiB, 69.35 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.24 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 120.99 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.12 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.89 W
0, 9 %, 70320 MiB, 132.65 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.18 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.69 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 120.28 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.17 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.77 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.14 W
7, 0 %, 4 MiB, 70.89 W
0, 5 %, 70320 MiB, 120.54 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.13 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.81 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.89 W
0, 5 %, 70320 MiB, 120.75 W
1, 0 %, 4 MiB, 68.71 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.34 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.83 W
6, 0 %, 4 MiB, 70.23 W
7, 0 %, 4 MiB, 70.87 W
0, 5 %, 70320 MiB, 121.14 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.22 W
7, 0 %, 4 MiB, 70.87 W
0, 5 %, 70320 MiB, 121.29 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.78 W
5, 0 %, 4 MiB, 68.72 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.59 W
1, 0 %, 4 MiB, 68.61 W
2, 0 %, 4 MiB, 69.20 W
3, 0 %, 4 MiB, 69.39 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.15 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.19 W
1, 0 %, 4 MiB, 68.63 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.88 W
0, 5 %, 70320 MiB, 121.38 W
1, 0 %, 4 MiB, 68.69 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.35 W
4, 0 %, 4 MiB, 69.75 W
5, 0 %, 4 MiB, 68.82 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 131.96 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.05 W
3, 0 %, 4 MiB, 69.35 W
4, 0 %, 4 MiB, 69.77 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.89 W
0, 5 %, 70320 MiB, 120.28 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.13 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.23 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 120.47 W
1, 0 %, 4 MiB, 68.66 W
2, 0 %, 4 MiB, 69.18 W
3, 0 %, 4 MiB, 69.35 W
4, 0 %, 4 MiB, 69.73 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.90 W
0, 5 %, 70320 MiB, 121.57 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.41 W
4, 0 %, 4 MiB, 69.73 W
5, 0 %, 4 MiB, 68.81 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.90 W
0, 5 %, 70320 MiB, 120.68 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.09 W
3, 0 %, 4 MiB, 69.38 W
4, 0 %, 4 MiB, 69.77 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.15 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.60 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.12 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.60 W
1, 0 %, 4 MiB, 68.75 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.87 W
0, 5 %, 70320 MiB, 121.42 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.19 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.73 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.91 W
0, 6 %, 70320 MiB, 130.18 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.18 W
3, 0 %, 4 MiB, 69.39 W
4, 0 %, 4 MiB, 69.79 W
5, 0 %, 4 MiB, 68.81 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.04 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.39 W
4, 0 %, 4 MiB, 69.75 W
5, 0 %, 4 MiB, 68.83 W
6, 0 %, 4 MiB, 70.22 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 120.76 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.16 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.75 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 120.49 W
1, 0 %, 4 MiB, 68.69 W
2, 0 %, 4 MiB, 69.20 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.87 W
0, 5 %, 70320 MiB, 122.09 W
1, 0 %, 4 MiB, 68.69 W
2, 0 %, 4 MiB, 69.15 W
3, 0 %, 4 MiB, 69.32 W
4, 0 %, 4 MiB, 69.78 W
5, 0 %, 4 MiB, 68.76 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.10 W
1, 0 %, 4 MiB, 68.69 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.35 W
4, 0 %, 4 MiB, 69.76 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.89 W
0, 5 %, 70320 MiB, 121.54 W
1, 0 %, 4 MiB, 68.72 W
2, 0 %, 4 MiB, 69.10 W
3, 0 %, 4 MiB, 69.38 W
4, 0 %, 4 MiB, 69.73 W
5, 0 %, 4 MiB, 68.84 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.93 W
0, 5 %, 70320 MiB, 121.15 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.12 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.86 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 120.84 W
1, 0 %, 4 MiB, 68.71 W
2, 0 %, 4 MiB, 69.21 W
3, 0 %, 4 MiB, 69.38 W
4, 0 %, 4 MiB, 69.78 W
5, 0 %, 4 MiB, 68.82 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.90 W
0, 5 %, 70320 MiB, 126.88 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.74 W
5, 0 %, 4 MiB, 68.81 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.32 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.13 W
3, 0 %, 4 MiB, 69.41 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.22 W
7, 0 %, 4 MiB, 70.90 W
0, 5 %, 70320 MiB, 120.07 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.08 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.74 W
5, 0 %, 4 MiB, 68.76 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.92 W
0, 5 %, 70320 MiB, 121.38 W
1, 0 %, 4 MiB, 68.60 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.80 W
5, 0 %, 4 MiB, 68.84 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.92 W
0, 5 %, 70320 MiB, 121.08 W
1, 0 %, 4 MiB, 68.67 W
2, 0 %, 4 MiB, 69.19 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.76 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.15 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 121.63 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.17 W
3, 0 %, 4 MiB, 69.39 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.93 W
0, 4 %, 70320 MiB, 120.09 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.15 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.74 W
5, 0 %, 4 MiB, 68.84 W
6, 0 %, 4 MiB, 70.24 W
7, 0 %, 4 MiB, 70.90 W
0, 4 %, 70320 MiB, 120.25 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.07 W
3, 0 %, 4 MiB, 69.36 W
4, 0 %, 4 MiB, 69.78 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.19 W
7, 0 %, 4 MiB, 70.91 W
0, 5 %, 70320 MiB, 120.55 W
1, 0 %, 4 MiB, 68.68 W
2, 0 %, 4 MiB, 69.09 W
3, 0 %, 4 MiB, 69.41 W
4, 0 %, 4 MiB, 69.76 W
5, 0 %, 4 MiB, 68.75 W
6, 0 %, 4 MiB, 70.25 W
7, 0 %, 4 MiB, 70.93 W
0, 4 %, 70320 MiB, 120.36 W
1, 0 %, 4 MiB, 68.69 W
2, 0 %, 4 MiB, 69.18 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.74 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.89 W
0, 4 %, 70320 MiB, 120.38 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.20 W
3, 0 %, 4 MiB, 69.39 W
4, 0 %, 4 MiB, 69.65 W
5, 0 %, 4 MiB, 68.83 W
6, 0 %, 4 MiB, 70.21 W
7, 0 %, 4 MiB, 70.92 W
0, 4 %, 70320 MiB, 120.29 W
1, 0 %, 4 MiB, 68.64 W
2, 0 %, 4 MiB, 69.14 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.74 W
5, 0 %, 4 MiB, 68.82 W
6, 0 %, 4 MiB, 70.17 W
7, 0 %, 4 MiB, 70.90 W
0, 6 %, 70320 MiB, 121.00 W
1, 0 %, 4 MiB, 68.65 W
2, 0 %, 4 MiB, 69.05 W
3, 0 %, 4 MiB, 69.33 W
4, 0 %, 4 MiB, 69.80 W
5, 0 %, 4 MiB, 68.79 W
6, 0 %, 4 MiB, 70.20 W
7, 0 %, 4 MiB, 70.93 W
0, 6 %, 70320 MiB, 121.38 W
1, 0 %, 4 MiB, 68.62 W
2, 0 %, 4 MiB, 69.11 W
3, 0 %, 4 MiB, 69.39 W
4, 0 %, 4 MiB, 69.77 W
5, 0 %, 4 MiB, 68.80 W
6, 0 %, 4 MiB, 70.24 W
7, 0 %, 4 MiB, 70.87 W
0, 6 %, 70320 MiB, 121.88 W
1, 0 %, 4 MiB, 68.70 W
2, 0 %, 4 MiB, 69.16 W
3, 0 %, 4 MiB, 69.37 W
4, 0 %, 4 MiB, 69.71 W
5, 0 %, 4 MiB, 68.71 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.94 W
0, 6 %, 70320 MiB, 122.93 W
1, 0 %, 4 MiB, 68.63 W
2, 0 %, 4 MiB, 69.18 W
3, 0 %, 4 MiB, 69.38 W
4, 0 %, 4 MiB, 69.72 W
5, 0 %, 4 MiB, 68.78 W
6, 0 %, 4 MiB, 70.18 W
7, 0 %, 4 MiB, 70.91 W
0, 6 %, 70320 MiB, 121.70 W
1, 0 %, 4 MiB, 68.75 W
2, 0 %, 4 MiB, 69.13 W
3, 0 %, 4 MiB, 69.40 W
4, 0 %, 4 MiB, 69.80 W
5, 0 %, 4 MiB, 68.84 W
6, 0 %, 4 MiB, 70.16 W
7, 0 %, 4 MiB, 70.91 W
```

## 17. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/server.log`

- 바이트 **50,564** · 줄 **362** · SHA256 `445e65d3ffa619e7ce6519310888334c55a1672ed6a0f6304501e28dc6157ce2`
- 인코딩 utf-8 · ANSI 시퀀스 2건 제거 · CR 9건 치환

```text
/sgl-workspace/sglang/python/sglang/launch_server.py:57: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
  Example: sglang serve --model-path <model> [options]
  warnings.warn(
'--disable-cuda-graph' is deprecated and will be removed in a future release. Use '--cuda-graph-backend-{decode,prefill}=disabled' instead.
[2026-08-27 03:51:48] server_args=ServerArgs(model_path='/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782', tokenizer_path='/models/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782', tokenizer_mode='auto', tokenizer_backend='huggingface', tokenizer_worker_num=1, detokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', model_config_parser='auto', json_model_override_args='{}', dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, enable_tf32_matmul=False, mem_fraction_static=0.841, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, retraction_policy='length', schedule_conservativeness=1.0, page_size=1, c128_page_size=16, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', prefill_only_disable_kv_cache=False, disable_radix_cache=False, enable_page_major_kv_layout=False, enable_unified_memory=False, disable_chunked_prefix_cache=False, disable_overlap_schedule=False, num_continuous_decode_steps=1, scheduler_recv_interval=1, enable_mixed_chunk=False, nccl_port=None, dist_timeout=None, dist_init_addr=None, nnodes=1, node_rank=0, tp_size=1, dcp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dwdp_size=1, dcp_comm_backend='ag_rs', dcp_replicate_q_proj=None, enable_prefill_cp=False, cp_strategy=None, enable_dsa_cache_layer_split=False, enable_dsa_prefill_context_parallel=False, dsa_prefill_cp_mode='round-robin-split', enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_cp_decode_attn_tp=False, enable_dp_attention=False, enable_dp_attention_local_control_broadcast=False, enable_dp_lm_head=False, enable_tp_lm_head_all_to_all=False, enable_attn_tp_input_scattered=False, disable_attn_tp_gather=False, enable_p2p_check=False, device='cuda', base_gpu_id=0, gpu_id_step=1, random_seed=138145041, mlx_enable_sampling=False, watchdog_timeout=300, soft_watchdog_timeout=None, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, numa_node=None, gc_threshold=None, host='127.0.0.1', port=30000, fastapi_root_path='', smg_grpc_mode=False, grpc_mode=False, grpc_port=None, sidecar=None, sidecar_args=None, skip_server_warmup=False, warmups=None, enable_http2=False, http2_max_concurrent_streams=200, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, api_key=None, admin_api_key=None, served_model_name='qwen30b', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, default_chat_template_kwargs=None, strip_thinking_cache=False, enable_strict_thinking=False, tool_call_parser=None, tool_server=None, sampling_defaults='model', asr_max_buffer_seconds=60, asr_max_concurrent_sessions=32, preferred_sampling_params=None, allow_auto_truncate=False, stream_interval=1, batch_notify_size=16, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, enable_session_radix_cache=False, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, smg_http_sidecar_port=None, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, load_snapshot_publish_interval=15, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_forward_pass_metrics=False, forward_pass_metrics_worker_id='', forward_pass_metrics_ipc_name=None, enable_trace=False, trace_modules='request', otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, stat_loggers=None, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', radix_cache_backend=None, mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', bf16_gemm_backend='auto', dsa_prefill_backend=None, dsa_decode_backend=None, dsa_paged_mqa_logits_backend='auto', dsa_topk_backend='sgl-kernel', disable_flashinfer_autotune=False, flashinfer_autotune_skip_ops=None, mamba_backend='triton', cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend='disabled', max_bs=256, bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], tc_compiler='eager', full_prefill_max_req=None, full_prefill_prefix_chunk_tokens=None), prefill=PhaseConfig(backend='disabled', max_bs=8192, bs=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], tc_compiler='eager', full_prefill_max_req=None, full_prefill_prefix_chunk_tokens=None)), cuda_graph_backend_decode=None, cuda_graph_backend_prefill=None, cuda_graph_max_bs_decode=None, cuda_graph_max_bs_prefill=None, cuda_graph_bs_decode=None, cuda_graph_bs_prefill=None, cuda_graph_tc_compiler=None, disable_prefill_cuda_graph=False, disable_decode_cuda_graph=False, disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, debug_cuda_graph=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, flashinfer_mla_disable_ragged=False, enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_deepseek_v4_fp4_indexer=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, enable_scattered_sconv=False, pre_warm_nccl=False, enable_quant_communications=False, enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, flashinfer_allreduce_fusion_backend=None, enable_aiter_allreduce_fusion=False, enable_torch_compile=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_dflash_block_size=None, speculative_dspark_block_size=None, speculative_dspark_sps_table_path=None, speculative_dspark_confidence_sts_path=None, speculative_dspark_align_verify_tokens_to_graph_tier=False, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_use_rejection_sampling=False, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_draft_kv_cache_dtype=None, speculative_draft_window_size=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, _speculative_draft_quantization_explicitly_set=False, speculative_skip_dp_mlp_sync=False, enable_multi_layer_eagle=False, speculative_adaptive=False, speculative_adaptive_config=None, decoupled_spec_bind_endpoint=None, decoupled_spec_connect_endpoints=None, decoupled_spec_rank=None, decoupled_spec_role='null', spec_trace_dir=None, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, speculative_ngram_external_corpus_path=None, speculative_ngram_external_sam_budget=0, speculative_ngram_external_corpus_max_tokens=10000000, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', deepep_mode='auto', fuseep_mode=2, deepep_dispatcher_output_dtype='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, expert_balancedness_report_mode='off', deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, enable_waterfill=False, ep_join_mode=None, ep_join_rank_offset=0, elastic_ep_initial_size=None, max_ep_size=None, elastic_ep_scale_timeout=600, elastic_ep_rejoin=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, disable_shared_experts_fusion=False, enforce_shared_experts_fusion=False, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_max_states_per_path=-1, enable_mamba_cache_stochastic_rounding=False, mamba_cache_philox_rounds=0, mamba_full_memory_ratio=0.9, mamba_radix_cache_strategy='auto', uses_mamba_radix_cache=False, mamba_track_interval=256, enable_int8_mamba_checkpoint=False, int8_mamba_ckpt_size=None, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, linear_attn_verify_backend=None, enable_linear_replayssm=False, linear_replayssm_cache_len=16, enable_linear_replayssm_spec=False, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='page_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='timeout', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, mm_processor_worker_num=0, mm_io_worker_num=0, allowed_media_domains=[], media_url_max_file_size_mb=64, mm_preprocess_cache_size_mb=None, trust_mm_content_hashes=False, limit_mm_data_per_request=None, enable_mm_global_cache=False, image_processor_backend='auto', mm_global_cache_backend='mooncake', disable_fast_image_processor=False, mm_feature_transport='cpu', keep_mm_feature_on_device=False, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, lora_use_virtual_experts=False, lora_strict_loading=False, lora_drain_wait_threshold=0.0, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', enable_lmcache=False, lmcache_config_file=None, enable_flexkv=False, flexkv_config_file=None, kt_weight_path='/models/kt/qwen3-30b-a3b-int8', kt_method='AMXINT8', kt_cpuinfer=96, kt_threadpool_count=2, kt_num_gpu_experts=0, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, dllm_fdfo=True, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_radix_cache=False, disaggregation_decode_enable_offload_kvcache=False, disaggregation_decode_retraction_backup=None, num_reserved_decode_tokens=512, disaggregation_decode_extra_slots=None, disaggregation_decode_polling_interval=1, optimistic_prefill_attempts=0, encoder_only=False, language_only=False, language_model_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], encoder_bootstrap_port=8997, encoder_register_urls=[], enable_adaptive_dispatch_to_encoder=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, startup_weight_load_mode='serial', custom_weight_loader=[], weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False, weight_loader_prefetch_num_threads=4, weight_loader_drop_cache_after_load=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, download_dir=None, model_checksum=None, delete_ckpt_after_loading=False, decrypted_config_file=None, decrypted_draft_config_file=None, checkpoint_engine_wait_weights_before_ready=False, enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, prefill_delayer_queue_min_ratio=None, prefill_delayer_max_delay_ms=None, min_free_slots_delay=None, enable_deterministic_inference=False, rl_on_policy_target=None, kv_canary='none', kv_canary_real_data='none', kv_canary_sweep_interval=0, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, enable_custom_logit_processor=False, enable_return_hidden_states=False, return_hidden_states_mode=None, enable_return_routed_experts=False, enable_return_indexer_topk=False, disable_outlines_disk_cache=False, enable_mis=False, weight_cache_mode='off', weight_cache_socket=None, weight_cache_timeout=1800, forward_hooks=None, msprobe_dump_config=None)
[2026-08-27 03:51:50] Using default HuggingFace chat template with detected content format: string
[2026-08-27 03:51:50] Auto-detected template features: reasoning_config=ReasoningToggleConfig(toggle_param='enable_thinking', default_enabled=True, special_case=None, effort_kwarg=None), reasoning_parser=qwen3, tool_call_parser=qwen
[2026-08-27 03:52:03] Init torch distributed begin.
[2026-08-27 03:52:03] Init torch distributed ends. elapsed=0.05 s, mem usage=0.09 GB
[2026-08-27 03:52:04] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 03:52:04] Load weight begin. avail mem=78.48 GB
[2026-08-27 03:52:04] Detected fp8 checkpoint.
[2026-08-27 03:52:04] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).

⏎ Multi-thread loading shards:   0% Completed | 0/7 [00:00<?, ?it/s]
⏎ Multi-thread loading shards:  14% Completed | 1/7 [00:02<00:12,  2.08s/it]
⏎ Multi-thread loading shards:  29% Completed | 2/7 [00:02<00:05,  1.02s/it]
⏎ Multi-thread loading shards:  43% Completed | 3/7 [00:02<00:02,  1.51it/s]
⏎ Multi-thread loading shards:  57% Completed | 4/7 [00:02<00:01,  2.03it/s]
⏎ Multi-thread loading shards:  71% Completed | 5/7 [00:03<00:00,  2.50it/s]
⏎ Multi-thread loading shards:  86% Completed | 6/7 [00:03<00:00,  2.89it/s]
⏎ Multi-thread loading shards: 100% Completed | 7/7 [00:03<00:00,  2.62it/s]
⏎ Multi-thread loading shards: 100% Completed | 7/7 [00:03<00:00,  1.86it/s]
CPUInfer[0x266028b0]: Hello
WorkerPool[0x254bf500] 2 subpools, [numa:threads][0:48] [1:48] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 48 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 48 threads
TP MOE layer 0, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 1, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 2, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 3, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 4, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 5, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 6, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 7, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 8, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 9, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 10, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 11, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 12, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 13, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 14, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 15, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 16, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 17, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 18, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 19, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 20, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 21, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 22, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 23, pool: 0x254bf500, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 24, pool: 0x254bf500, expert nu[2026-08-27 03:52:31] Load weight end. elapsed=26.91 s, type=Qwen3MoeForCausalLM, quant=fp8, fmt=e4m3, avail mem=76.34 GB, mem usage=2.14 GB.
[2026-08-27 03:52:32] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 697336, K size: 31.92 GB, V size: 31.92 GB
[2026-08-27 03:52:32] Memory pool end. avail mem=11.77 GB
[2026-08-27 03:52:32] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 03:52:32] max_total_num_tokens=697336, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=40960, available_gpu_mem=11.64 GB
[2026-08-27 03:52:32] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 03:52:33] Engine startup timings (s): load_weight=26.91, kv_cache_allocation=0.03, scheduler_e2e=31.39, cuda_graph={prefill=0.00, decode=0.00, target_verify=0.00, draft_prefill=0.00, draft_decode=0.00, draft_extend=0.00}, tokenizer_e2e=45.21
[2026-08-27 03:52:33] INFO:     Started server process [8884]
[2026-08-27 03:52:33] INFO:     Waiting for application startup.
[2026-08-27 03:52:33] Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}
[2026-08-27 03:52:33] INFO:     Application startup complete.
[2026-08-27 03:52:33] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2026-08-27 03:52:34] INFO:     127.0.0.1:37200 - "GET /model_info HTTP/1.1" 200 OK
[2026-08-27 03:52:35] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=5120,K=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 03:52:35] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2048,K=4096,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 03:52:35] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=768,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 03:52:35] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=768,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 03:52:37] INFO:     127.0.0.1:37208 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 03:52:37] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 1.19
[2026-08-27 03:52:38] INFO:     127.0.0.1:37204 - "POST /generate HTTP/1.1" 200 OK
[2026-08-27 03:52:38] Freezing GC in Scheduler process. gen0: 48->0, gen1: 20->0, gen2: 1087627->0
[2026-08-27 03:52:38] Freezing GC in Tokenizer Manager process. gen0: 334->0, gen1: 5166->0, gen2: 912143->0
[2026-08-27 03:52:38] INFO:     127.0.0.1:37210 - "POST /freeze_gc HTTP/1.1" 200 OK
[2026-08-27 03:52:38] The server is fired up and ready to roll!
[2026-08-27 03:52:39] Freezing GC in Detokenizer Manager process. gen0: 382->0, gen1: 0->0, gen2: 851094->0
[2026-08-27 03:52:42] Prefill batch, #new-seq: 1, #new-token: 1, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.21
[2026-08-27 03:52:43] INFO:     127.0.0.1:42722 - "GET /health HTTP/1.1" 200 OK
[2026-08-27 03:52:43] Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 2, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 2.60
[2026-08-27 03:52:48] Decode batch, #running-req: 1, #token: 0, token usage: 0.00, cuda graph: False, gen throughput (token/s): 2.62, #queue-req: 0
[2026-08-27 03:52:48] INFO:     127.0.0.1:42736 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:03] INFO:     127.0.0.1:54144 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 03:53:03] INFO:     127.0.0.1:54144 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 03:53:05] Prefill batch, #new-seq: 1, #new-token: 1018, #cached-token: 0, token usage: 0.01, #running-req: 0, #queue-req: 6, #pending-token: 0, cuda graph: False, input throughput (token/s): 46.18
[2026-08-27 03:53:05] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] Prefill batch, #new-seq: 9, #new-token: 8192, #cached-token: 0, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 6993, cuda graph: False, input throughput (token/s): 22672.00
[2026-08-27 03:53:06] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:06] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:07] Prefill batch, #new-seq: 7, #new-token: 5787, #cached-token: 1206, token usage: 0.02, #running-req: 9, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 4661.61
[2026-08-27 03:53:07] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:07] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:07] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:07] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:07] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:07] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:07] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:13] Decode batch, #running-req: 16, #token: 13814, token usage: 0.02, cuda graph: False, gen throughput (token/s): 24.75, #queue-req: 0
[2026-08-27 03:53:19] Decode batch, #running-req: 16, #token: 14454, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.21, #queue-req: 0
[2026-08-27 03:53:25] Decode batch, #running-req: 16, #token: 15094, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.48, #queue-req: 0
[2026-08-27 03:53:31] Decode batch, #running-req: 16, #token: 15734, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.70, #queue-req: 0
[2026-08-27 03:53:38] Decode batch, #running-req: 16, #token: 16374, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.50, #queue-req: 0
[2026-08-27 03:53:44] Decode batch, #running-req: 16, #token: 17014, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.25, #queue-req: 0
[2026-08-27 03:53:47] Prefill batch, #new-seq: 1, #new-token: 823, #cached-token: 201, token usage: 0.01, #running-req: 0, #queue-req: 4, #pending-token: 0, cuda graph: False, input throughput (token/s): 20.51
[2026-08-27 03:53:47] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] Prefill batch, #new-seq: 11, #new-token: 8192, #cached-token: 2235, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 4756, cuda graph: False, input throughput (token/s): 4634.75
[2026-08-27 03:53:49] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] Prefill batch, #new-seq: 5, #new-token: 3952, #cached-token: 804, token usage: 0.02, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 5150.10
[2026-08-27 03:53:49] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:49] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:53:53] Decode batch, #running-req: 16, #token: 13575, token usage: 0.02, cuda graph: False, gen throughput (token/s): 69.52, #queue-req: 0
[2026-08-27 03:53:59] Decode batch, #running-req: 16, #token: 14215, token usage: 0.02, cuda graph: False, gen throughput (token/s): 104.20, #queue-req: 0
[2026-08-27 03:54:05] Decode batch, #running-req: 16, #token: 14855, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.86, #queue-req: 0
[2026-08-27 03:54:12] Decode batch, #running-req: 16, #token: 15495, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.83, #queue-req: 0
[2026-08-27 03:54:18] Decode batch, #running-req: 16, #token: 16135, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.23, #queue-req: 0
[2026-08-27 03:54:24] Decode batch, #running-req: 16, #token: 16775, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.09, #queue-req: 0
[2026-08-27 03:54:30] Prefill batch, #new-seq: 1, #new-token: 815, #cached-token: 202, token usage: 0.01, #running-req: 0, #queue-req: 4, #pending-token: 0, cuda graph: False, input throughput (token/s): 20.33
[2026-08-27 03:54:30] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] Prefill batch, #new-seq: 11, #new-token: 8192, #cached-token: 2214, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 4812, cuda graph: False, input throughput (token/s): 4767.47
[2026-08-27 03:54:31] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:31] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:32] Prefill batch, #new-seq: 5, #new-token: 4008, #cached-token: 804, token usage: 0.02, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 5185.28
[2026-08-27 03:54:32] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:32] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:32] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:32] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:32] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:54:33] Decode batch, #running-req: 16, #token: 13345, token usage: 0.02, cuda graph: False, gen throughput (token/s): 69.70, #queue-req: 0
[2026-08-27 03:54:39] Decode batch, #running-req: 16, #token: 13985, token usage: 0.02, cuda graph: False, gen throughput (token/s): 104.22, #queue-req: 0
[2026-08-27 03:54:45] Decode batch, #running-req: 16, #token: 14625, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.89, #queue-req: 0
[2026-08-27 03:54:52] Decode batch, #running-req: 16, #token: 15265, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.25, #queue-req: 0
[2026-08-27 03:54:58] Decode batch, #running-req: 16, #token: 15905, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.56, #queue-req: 0
[2026-08-27 03:55:04] Decode batch, #running-req: 16, #token: 16545, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.35, #queue-req: 0
[2026-08-27 03:55:10] Decode batch, #running-req: 16, #token: 17185, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.85, #queue-req: 0
[2026-08-27 03:55:12] Prefill batch, #new-seq: 1, #new-token: 818, #cached-token: 201, token usage: 0.01, #running-req: 0, #queue-req: 4, #pending-token: 0, cuda graph: False, input throughput (token/s): 20.35
[2026-08-27 03:55:12] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] Prefill batch, #new-seq: 11, #new-token: 8192, #cached-token: 2215, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 4873, cuda graph: False, input throughput (token/s): 4785.11
[2026-08-27 03:55:14] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:14] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:15] Prefill batch, #new-seq: 5, #new-token: 4056, #cached-token: 817, token usage: 0.02, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 5186.13
[2026-08-27 03:55:15] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:15] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:15] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:15] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:15] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:20] Decode batch, #running-req: 16, #token: 13795, token usage: 0.02, cuda graph: False, gen throughput (token/s): 69.52, #queue-req: 0
[2026-08-27 03:55:26] Decode batch, #running-req: 16, #token: 14435, token usage: 0.02, cuda graph: False, gen throughput (token/s): 104.03, #queue-req: 0
[2026-08-27 03:55:32] Decode batch, #running-req: 16, #token: 15075, token usage: 0.02, cuda graph: False, gen throughput (token/s): 104.14, #queue-req: 0
[2026-08-27 03:55:38] Decode batch, #running-req: 16, #token: 15715, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.54, #queue-req: 0
[2026-08-27 03:55:44] Decode batch, #running-req: 16, #token: 16355, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.78, #queue-req: 0
[2026-08-27 03:55:50] Decode batch, #running-req: 16, #token: 16995, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.45, #queue-req: 0
[2026-08-27 03:55:55] Prefill batch, #new-seq: 1, #new-token: 796, #cached-token: 201, token usage: 0.01, #running-req: 0, #queue-req: 4, #pending-token: 0, cuda graph: False, input throughput (token/s): 19.88
[2026-08-27 03:55:55] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] Prefill batch, #new-seq: 11, #new-token: 8192, #cached-token: 2238, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 4766, cuda graph: False, input throughput (token/s): 4745.81
[2026-08-27 03:55:56] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:56] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:57] Prefill batch, #new-seq: 5, #new-token: 3960, #cached-token: 806, token usage: 0.02, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 5108.85
[2026-08-27 03:55:57] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:57] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:57] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:57] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:55:57] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:00] Decode batch, #running-req: 16, #token: 13421, token usage: 0.02, cuda graph: False, gen throughput (token/s): 70.16, #queue-req: 0
[2026-08-27 03:56:06] Decode batch, #running-req: 16, #token: 14061, token usage: 0.02, cuda graph: False, gen throughput (token/s): 104.31, #queue-req: 0
[2026-08-27 03:56:12] Decode batch, #running-req: 16, #token: 14701, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.36, #queue-req: 0
[2026-08-27 03:56:18] Decode batch, #running-req: 16, #token: 15341, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.77, #queue-req: 0
[2026-08-27 03:56:24] Decode batch, #running-req: 16, #token: 15981, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.06, #queue-req: 0
[2026-08-27 03:56:31] Decode batch, #running-req: 16, #token: 16621, token usage: 0.02, cuda graph: False, gen throughput (token/s): 101.02, #queue-req: 0
[2026-08-27 03:56:37] Decode batch, #running-req: 16, #token: 0, token usage: 0.00, cuda graph: False, gen throughput (token/s): 102.80, #queue-req: 0
[2026-08-27 03:56:37] Prefill batch, #new-seq: 1, #new-token: 799, #cached-token: 202, token usage: 0.01, #running-req: 0, #queue-req: 4, #pending-token: 0, cuda graph: False, input throughput (token/s): 19.88
[2026-08-27 03:56:37] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] Prefill batch, #new-seq: 11, #new-token: 8192, #cached-token: 2229, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 4760, cuda graph: False, input throughput (token/s): 4802.76
[2026-08-27 03:56:39] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:39] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:40] Prefill batch, #new-seq: 5, #new-token: 3954, #cached-token: 806, token usage: 0.02, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 5172.72
[2026-08-27 03:56:40] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:40] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:40] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:40] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:40] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:56:46] Decode batch, #running-req: 16, #token: 13803, token usage: 0.02, cuda graph: False, gen throughput (token/s): 70.64, #queue-req: 0
[2026-08-27 03:56:52] Decode batch, #running-req: 16, #token: 14443, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.29, #queue-req: 0
[2026-08-27 03:56:58] Decode batch, #running-req: 16, #token: 15083, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.80, #queue-req: 0
[2026-08-27 03:57:05] Decode batch, #running-req: 16, #token: 15723, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.76, #queue-req: 0
[2026-08-27 03:57:11] Decode batch, #running-req: 16, #token: 16363, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.71, #queue-req: 0
[2026-08-27 03:57:17] Decode batch, #running-req: 16, #token: 17003, token usage: 0.02, cuda graph: False, gen throughput (token/s): 102.79, #queue-req: 0
[2026-08-27 03:57:20] Prefill batch, #new-seq: 1, #new-token: 809, #cached-token: 202, token usage: 0.01, #running-req: 0, #queue-req: 4, #pending-token: 0, cuda graph: False, input throughput (token/s): 20.13
[2026-08-27 03:57:20] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] Prefill batch, #new-seq: 11, #new-token: 8192, #cached-token: 2243, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 4803, cuda graph: False, input throughput (token/s): 4790.55
[2026-08-27 03:57:22] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:22] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:23] Prefill batch, #new-seq: 5, #new-token: 3990, #cached-token: 813, token usage: 0.02, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 5167.76
[2026-08-27 03:57:23] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:23] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:23] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:23] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:23] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:57:26] Decode batch, #running-req: 16, #token: 13607, token usage: 0.02, cuda graph: False, gen throughput (token/s): 70.24, #queue-req: 0
[2026-08-27 03:57:32] Decode batch, #running-req: 16, #token: 14247, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.62, #queue-req: 0
[2026-08-27 03:57:38] Decode batch, #running-req: 16, #token: 14887, token usage: 0.02, cuda graph: False, gen throughput (token/s): 103.19, #queue-req: 0
[2026-08-27 03:57:44] Decode batch, #running-req: 16, #token: 15527, token usage: 0.02, cuda graph: False, gen throughput (token/s): 107.31, #queue-req: 0
[2026-08-27 03:57:50] Decode batch, #running-req: 16, #token: 16167, token usage: 0.02, cuda graph: False, gen throughput (token/s): 108.51, #queue-req: 0
[2026-08-27 03:57:58] Decode batch, #running-req: 16, #token: 16807, token usage: 0.02, cuda graph: False, gen throughput (token/s): 89.44, #queue-req: 0
[2026-08-27 03:58:04] Prefill batch, #new-seq: 1, #new-token: 809, #cached-token: 209, token usage: 0.01, #running-req: 0, #queue-req: 4, #pending-token: 0, cuda graph: False, input throughput (token/s): 19.60
[2026-08-27 03:58:04] INFO:     127.0.0.1:54268 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] Prefill batch, #new-seq: 11, #new-token: 8192, #cached-token: 2231, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 4764, cuda graph: False, input throughput (token/s): 4780.82
[2026-08-27 03:58:06] INFO:     127.0.0.1:54228 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54246 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54198 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54164 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54208 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54264 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54176 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54224 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54256 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54154 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] Prefill batch, #new-seq: 5, #new-token: 3933, #cached-token: 831, token usage: 0.02, #running-req: 11, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 5153.23
[2026-08-27 03:58:06] INFO:     127.0.0.1:54144 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54190 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54152 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54226 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:06] INFO:     127.0.0.1:54234 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 03:58:08] Decode batch, #running-req: 16, #token: 13317, token usage: 0.02, cuda graph: False, gen throughput (token/s): 63.57, #queue-req: 0
[2026-08-27 03:58:15] Decode batch, #running-req: 16, #token: 13957, token usage: 0.02, cuda graph: False, gen throughput (token/s): 89.69, #queue-req: 0
[2026-08-27 03:58:21] Decode batch, #running-req: 16, #token: 14597, token usage: 0.02, cuda graph: False, gen throughput (token/s): 94.28, #queue-req: 0
[2026-08-27 03:58:27] Decode batch, #running-req: 16, #token: 15237, token usage: 0.02, cuda graph: False, gen throughput (token/s): 116.88, #queue-req: 0
[2026-08-27 03:58:32] Decode batch, #running-req: 16, #token: 15877, token usage: 0.02, cuda graph: False, gen throughput (token/s): 119.05, #queue-req: 0
[2026-08-27 03:58:38] Decode batch, #running-req: 16, #token: 16517, token usage: 0.02, cuda graph: False, gen throughput (token/s): 122.02, #queue-req: 0
[2026-08-27 03:58:43] Decode batch, #running-req: 16, #token: 17157, token usage: 0.02, cuda graph: False, gen throughput (token/s): 123.88, #queue-req: 0
[2026-08-27 03:58:44] INFO:     127.0.0.1:54268 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 03:58:44] INFO:     127.0.0.1:54228 - "GET /metrics HTTP/1.1" 404 Not Found
```

## 18. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/server_fail.log`

- 바이트 **2,703** · 줄 **60** · SHA256 `b9778e075146e21a3c5702cfe881b06fb8ca7ccae5b37b055785610750441463`
- 인코딩 utf-8

```text
TP Load from loader
TP MOE layer 34, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 35, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 36, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 37, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 38, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 39, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 40, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 41, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 42, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 43, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 44, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 45, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 46, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 47, pool: 0x24a112b0, expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
[rank0]:[W827 03:49:52.048426227 ProcessGroupNCCL.cpp:1624] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[2026-08-27 03:49:55] No live scheduler processes found; skipping py-spy and CUDA coredump.
[2026-08-27 03:49:55] kill_process_tree called: parent_pid=7680, include_parent=True, pid=7680
```

## 19. `eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/smoke.json`

- 바이트 **509** · 줄 **1** · SHA256 `24a3bf847be22d787102672fbe7c61b976f6334ec46ec81b09af794109521ec1`
- 인코딩 utf-8

정리한 형태:

```json
{
  "id": "dbe781967a6240b0858abd9dfda495ec",
  "object": "text_completion",
  "created": 1787802768,
  "model": "qwen30b",
  "choices": [
    {
      "index": 0,
      "text": " Paris. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital of Spain is Madrid. The capital of Italy is Rome.",
      "logprobs": null,
      "finish_reason": "length",
      "matched_stop": null
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 37,
    "completion_tokens": 32,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

원문:

```text
{"id":"dbe781967a6240b0858abd9dfda495ec","object":"text_completion","created":1787802768,"model":"qwen30b","choices":[{"index":0,"text":" Paris. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital of Spain is Madrid. The capital of Italy is Rome.","logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":5,"total_tokens":37,"completion_tokens":32,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
```

# 원시 파일 — `eval/results/20260827_133055_tsk043_main_r1/`

R1-0528 본판 1차 — r0 GPU-only OOM 실증 / r1 FP8 직독 실패

## 20. `eval/results/20260827_133055_tsk043_main_r1/RUN.log`

- 바이트 **279** · 줄 **7** · SHA256 `b292f7f0b035a14e0f000ce6677aea0ff0d4e9675a04617ae8674a7549954b05`
- 인코딩 utf-8

```text
== TSK_043 main start 20260827_133055 model=/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 ==
-- r0_gpu_only_oom --
r0 verdict=DIED_AS_EXPECTED after 90s
-- r1_kt_hybrid --
SGL DIED 90s
r1 FAIL
== TSK_043 main done 133415 ==
```

## 21. `eval/results/20260827_133055_tsk043_main_r1/r0_gpu_only_oom/oom_evidence.log`

- 바이트 **602** · 줄 **5** · SHA256 `77d2a9f7725ba85c4b8e4ec763360cdf36ffa31fcfb8fe9748744de2d1c9a582`
- 인코딩 utf-8 · CR 2건 치환

```text
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 805, in wait_for_ready
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     raise e from None
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346] Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
(APIServer pid=42239)     raise RuntimeError(
(APIServer pid=42239) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

## 22. `eval/results/20260827_133055_tsk043_main_r1/r0_gpu_only_oom/server.log`

- 바이트 **112,499** · 줄 **737** · SHA256 `618fad8d459f9c17e411c14dadc605891b956cf4fec6d9106718817573b53fd5`
- 인코딩 utf-8 · CR 574건 치환

```text
(APIServer pid=42239) INFO 08-27 04:31:07 [api_utils.py:333] 
(APIServer pid=42239) INFO 08-27 04:31:07 [api_utils.py:333]        █     █     █▄   ▄█
(APIServer pid=42239) INFO 08-27 04:31:07 [api_utils.py:333]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.28.0
(APIServer pid=42239) INFO 08-27 04:31:07 [api_utils.py:333]   █▄█▀ █     █     █     █  model   /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
(APIServer pid=42239) INFO 08-27 04:31:07 [api_utils.py:333]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=42239) INFO 08-27 04:31:07 [api_utils.py:333] 
(APIServer pid=42239) INFO 08-27 04:31:07 [api_utils.py:272] non-default args: {'model_tag': '/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52', 'host': '127.0.0.1', 'model': '/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52', 'max_model_len': 8192, 'served_model_name': ['r1'], 'tensor_parallel_size': 8, 'gpu_memory_utilization': 0.95}
(APIServer pid=42239) INFO 08-27 04:31:24 [model.py:672] Resolved architecture: DeepseekV3ForCausalLM
(APIServer pid=42239) INFO 08-27 04:31:24 [model.py:1965] Using max model len 8192
(APIServer pid=42239) INFO 08-27 04:31:29 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.
(APIServer pid=42239) INFO 08-27 04:31:29 [kernel.py:308] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(APIServer pid=42239) INFO 08-27 04:31:29 [compilation.py:329] Enabled custom fusions: norm_quant, act_quant, allreduce_rms
(EngineCore pid=42815) INFO 08-27 04:31:44 [core.py:122] Initializing a V1 LLM engine (v0.28.0) with config: model='/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52', speculative_config=None, tokenizer='/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=8, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=fp8, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=r1, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['+quant_fp8', 'none', '+quant_fp8'], 'ir_enable_torch_wrap': True, 'splitting_ops': ['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::qwen_gdn_attention_core', 'vllm::qwen_gdn_attention_core_fused_norm_packed', 'vllm::gdn_attention_core_xpu', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', 'vllm::hpc_rope_norm_forward', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [36, 8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': True, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, enable_jit_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')
(EngineCore pid=42815) INFO 08-27 04:31:44 [multiproc_executor.py:150] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=211.115.75.81 (local), world_size=8, local_world_size=8
(Worker pid=43055) INFO 08-27 04:32:01 [parallel_state.py:1638] world_size=8 rank=2 local_rank=2 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43053) INFO 08-27 04:32:02 [parallel_state.py:1638] world_size=8 rank=0 local_rank=0 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43058) INFO 08-27 04:32:02 [parallel_state.py:1638] world_size=8 rank=5 local_rank=5 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43060) INFO 08-27 04:32:02 [parallel_state.py:1638] world_size=8 rank=7 local_rank=7 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43059) INFO 08-27 04:32:03 [parallel_state.py:1638] world_size=8 rank=6 local_rank=6 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43054) INFO 08-27 04:32:03 [parallel_state.py:1638] world_size=8 rank=1 local_rank=1 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43057) INFO 08-27 04:32:03 [parallel_state.py:1638] world_size=8 rank=4 local_rank=4 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43056) INFO 08-27 04:32:03 [parallel_state.py:1638] world_size=8 rank=3 local_rank=3 distributed_init_method=file:///tmp/vllm_dist_26d343a436e54302b7ad9bda252b8f1a backend=nccl
(Worker pid=43053) INFO 08-27 04:32:03 [pynccl.py:113] vLLM is using nccl==2.30.7
(Worker pid=43053) INFO 08-27 04:32:06 [cuda_communicator.py:266] Using ['CUSTOM', 'SYMM_MEM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=43053) INFO 08-27 04:32:08 [cuda_communicator.py:266] Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'ep:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=43053) INFO 08-27 04:32:08 [parallel_state.py:1982] rank 0 in world size 8 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
(Worker pid=43053) INFO 08-27 04:32:08 [topk_topp_sampler.py:62] Using FlashInfer for top-p & top-k sampling.
(Worker_TP0 pid=43053) INFO 08-27 04:32:08 [gpu_model_runner.py:5419] Starting to load model /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52...
(Worker_TP0 pid=43053) INFO 08-27 04:32:09 [__init__.py:689] Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8LinearMethod
(Worker_TP0 pid=43053) INFO 08-27 04:32:09 [deep_gemm.py:186] deep_gemm not found in site-packages, trying vendored vllm.third_party.deep_gemm
(Worker_TP0 pid=43053) INFO 08-27 04:32:09 [deep_gemm.py:213] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
(Worker_TP0 pid=43053) INFO 08-27 04:32:09 [deep_gemm.py:131] DeepGEMM E8M0 enabled on current platform.
(Worker_TP0 pid=43053) INFO 08-27 04:32:09 [cuda.py:486] Using FLASH_ATTN_MLA attention backend out of potential backends: ['FLASH_ATTN_MLA', 'FLASHMLA', 'TRITON_MLA'].
(Worker_TP0 pid=43053) INFO 08-27 04:32:09 [selector.py:190] Using FLASH_ATTN MLA prefill backend.
(Worker_TP0 pid=43053) INFO 08-27 04:32:09 [fp8.py:411] Using TRITON Fp8 MoE backend out of potential backends: ['TRITON', 'AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS', 'DEEPGEMM', 'MARLIN', 'HUMMING', 'BATCHED_DEEPGEMM', 'BATCHED_TRITON', 'XPU', 'CPU', 'HPC'].
[rank7]:[W827 04:32:09.535618155 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 7 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank7]:[W827 04:32:09.537898180 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 7 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
[rank5]:[W827 04:32:09.546394232 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 5 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank5]:[W827 04:32:09.547572606 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 5 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
[rank6]:[W827 04:32:09.551084160 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 6 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank0]:[W827 04:32:09.551099660 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank6]:[W827 04:32:09.552894862 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 6 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
[rank0]:[W827 04:32:09.552935651 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
[rank1]:[W827 04:32:09.582350121 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 1 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank1]:[W827 04:32:09.583059757 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 1 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
[rank3]:[W827 04:32:09.588787136 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 3 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank4]:[W827 04:32:09.589385298 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 4 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank3]:[W827 04:32:09.589754488 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 3 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
[rank4]:[W827 04:32:09.592216057 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 4 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
[rank2]:[W827 04:32:09.625174322 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 2 while trying to allocate 939524096 bytes (free: 301989888, total: 85017493504).
[rank2]:[W827 04:32:09.625876159 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 2 while trying to allocate 939524096 bytes (free: 364904448, total: 85017493504).
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 5 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5 pid=43058) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 5 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
(EngineCore pid=42815) INFO 08-27 04:32:10 [multiproc_executor.py:469] [shutdown] Executor: waiting for worker exit count=8
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 6 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6 pid=43059) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 6 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 7 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 1 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP7 pid=43060) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 7 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=43054) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 1 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 3 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=43053) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP3 pid=43056) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 3 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 4 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP4 pid=43057) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 4 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [gpu_model_runner.py:5513] Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips. (original error: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 2 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf))
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941] WorkerProc failed to start.
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941] Traceback (most recent call last):
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 908, in worker_main
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     worker = WorkerProc(*args, **kwargs)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 677, in __init__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.worker.load_model()
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 457, in load_model
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5514, in load_model
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     raise e
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5435, in load_model
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = model_loader.load_model(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 55, in load_model
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = initialize_model(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 58, in initialize_model
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     model = model_class(vllm_config=vllm_config, prefix=prefix)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1836, in __init__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.model = self.model_cls(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                  ^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 383, in __init__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     old_init(self, *args, **kwargs)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1394, in __init__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.start_layer, self.end_layer, self.layers = make_layers(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                                                     ^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 812, in make_layers
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     + get_offloader().wrap_modules(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/offloader/base.py", line 104, in wrap_modules
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return list(modules_generator)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 813, in <genexpr>
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1396, in <lambda>
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     lambda prefix: DeepseekV2DecoderLayer(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1264, in __init__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.mlp = DeepseekV2MoE(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                ^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 358, in __init__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.experts = FusedMoEFactory(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                    ^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 375, in FusedMoEFactory
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     routed_experts = routed_experts_cls(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]                      ^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/routed_experts.py", line 175, in __init__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py", line 550, in create_weights
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     torch.empty(
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_device.py", line 122, in __torch_function__
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]     return func(*args, **kwargs)
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=43055) ERROR 08-27 04:32:10 [multiproc_executor.py:941] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 2 has a total capacity of 79.18 GiB of which 348.00 MiB is free. Including non-PyTorch memory, this process has 78.83 GiB memory in use. Of the allocated memory 76.62 GiB is allocated by PyTorch, and 73.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
[rank0]:[W827 04:32:11.405900564 ProcessGroupNCCL.cpp:1624] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
(EngineCore pid=42815) INFO 08-27 04:32:14 [multiproc_executor.py:476] [shutdown] Executor: all workers exited gracefully
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346] EngineCore failed to start.
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346] Traceback (most recent call last):
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1315, in run_engine_core
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     return func(*args, **kwargs)
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1073, in __init__
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     super().__init__(
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 133, in __init__
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     self.model_executor = executor_class(vllm_config)
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 116, in __init__
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     super().__init__(vllm_config)
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     return func(*args, **kwargs)
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 110, in __init__
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     self._init_executor()
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 211, in _init_executor
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     self.workers = WorkerProc.wait_for_ready(unready_workers)
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 805, in wait_for_ready
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346]     raise e from None
(EngineCore pid=42815) ERROR 08-27 04:32:14 [core.py:1346] Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
(EngineCore pid=42815) Process EngineCore:
(EngineCore pid=42815) Traceback (most recent call last):
(EngineCore pid=42815)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore pid=42815)     self.run()
(EngineCore pid=42815)   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore pid=42815)     self._target(*self._args, **self._kwargs)
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1350, in run_engine_core
(EngineCore pid=42815)     raise e
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1315, in run_engine_core
(EngineCore pid=42815)     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=42815)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=42815)     return func(*args, **kwargs)
(EngineCore pid=42815)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1073, in __init__
(EngineCore pid=42815)     super().__init__(
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 133, in __init__
(EngineCore pid=42815)     self.model_executor = executor_class(vllm_config)
(EngineCore pid=42815)                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 116, in __init__
(EngineCore pid=42815)     super().__init__(vllm_config)
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=42815)     return func(*args, **kwargs)
(EngineCore pid=42815)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 110, in __init__
(EngineCore pid=42815)     self._init_executor()
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 211, in _init_executor
(EngineCore pid=42815)     self.workers = WorkerProc.wait_for_ready(unready_workers)
(EngineCore pid=42815)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=42815)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 805, in wait_for_ready
(EngineCore pid=42815)     raise e from None
(EngineCore pid=42815) Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
(APIServer pid=42239) INFO 08-27 04:32:16 [utils.py:615] [shutdown] Process manager: send sigterm to process EngineCore
(APIServer pid=42239) Traceback (most recent call last):
(APIServer pid=42239)   File "/usr/local/bin/vllm", line 10, in <module>
(APIServer pid=42239)     sys.exit(main())
(APIServer pid=42239)              ^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/main.py", line 97, in main
(APIServer pid=42239)     args.dispatch_function(args)
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/serve.py", line 152, in cmd
(APIServer pid=42239)     uvloop.run(run_server(args))
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 96, in run
(APIServer pid=42239)     return __asyncio.run(
(APIServer pid=42239)            ^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
(APIServer pid=42239)     return runner.run(main)
(APIServer pid=42239)            ^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
(APIServer pid=42239)     return self._loop.run_until_complete(task)
(APIServer pid=42239)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 48, in wrapper
(APIServer pid=42239)     return await main
(APIServer pid=42239)            ^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 739, in run_server
(APIServer pid=42239)     await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 753, in run_server_worker
(APIServer pid=42239)     async with build_async_engine_client(
(APIServer pid=42239)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=42239)     return await anext(self.gen)
(APIServer pid=42239)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 127, in build_async_engine_client
(APIServer pid=42239)     async with build_async_engine_client_from_engine_args(
(APIServer pid=42239)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=42239)     return await anext(self.gen)
(APIServer pid=42239)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 163, in build_async_engine_client_from_engine_args
(APIServer pid=42239)     async_llm = AsyncLLM.from_vllm_config(
(APIServer pid=42239)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 220, in from_vllm_config
(APIServer pid=42239)     return cls(
(APIServer pid=42239)            ^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 149, in __init__
(APIServer pid=42239)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=42239)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=42239)     return func(*args, **kwargs)
(APIServer pid=42239)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 139, in make_async_mp_client
(APIServer pid=42239)     return AsyncMPClient(*client_args)
(APIServer pid=42239)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=42239)     return func(*args, **kwargs)
(APIServer pid=42239)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 990, in __init__
(APIServer pid=42239)     super().__init__(
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 609, in __init__
(APIServer pid=42239)     with launch_core_engines(
(APIServer pid=42239)   File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=42239)     next(self.gen)
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 1206, in launch_core_engines
(APIServer pid=42239)     wait_for_engine_startup(
(APIServer pid=42239)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 1286, in wait_for_engine_startup
(APIServer pid=42239)     raise RuntimeError(
(APIServer pid=42239) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

## 23. `eval/results/20260827_133055_tsk043_main_r1/r1_kt_hybrid/server_fail.log`

- 바이트 **18,872** · 줄 **195** · SHA256 `f4297511781379f7a113564779975025a1b71e7c10812b99153169f6e8719ee4`
- 인코딩 utf-8 · CR 75건 치환

```text
[2026-08-27 04:32:58] Auto-detected template features: reasoning_parser=deepseek-r1, tool_call_parser=deepseekv31
[2026-08-27 04:33:07 TP0] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:07 TP0] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:08 TP0] Init torch distributed begin.
[2026-08-27 04:33:08 TP2] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:08 TP2] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:09 TP2] Init torch distributed begin.
[2026-08-27 04:33:09 TP3] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:09 TP3] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:09 TP1] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:09 TP1] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:09 TP3] Init torch distributed begin.
[2026-08-27 04:33:09 TP4] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:09 TP4] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:10 TP1] Init torch distributed begin.
[2026-08-27 04:33:10 TP4] Init torch distributed begin.
[2026-08-27 04:33:10 TP5] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:10 TP5] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:10 TP5] Init torch distributed begin.
[2026-08-27 04:33:10] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:10] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:10 TP6] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:10 TP6] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:11 TP6] Init torch distributed begin.
[2026-08-27 04:33:11 TP7] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 04:33:11 TP7] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 04:33:11 TP7] Init torch distributed begin.
[2026-08-27 04:33:11 TP0] sglang is using nccl==2.29.7
[2026-08-27 04:33:24 TP0] All Reduce config: symmetric_memory = 18.01 MB, local_buffer = 8.00 MB, multicast = True
[2026-08-27 04:33:34 TP0] Init torch distributed ends. elapsed=26.80 s, mem usage=1.25 GB
[2026-08-27 04:33:34 TP7] Init torch distributed ends. elapsed=23.02 s, mem usage=1.06 GB
[2026-08-27 04:33:34 TP6] Init torch distributed ends. elapsed=23.66 s, mem usage=1.29 GB
[2026-08-27 04:33:34 TP5] Init torch distributed ends. elapsed=24.20 s, mem usage=1.29 GB
[2026-08-27 04:33:34 TP4] Init torch distributed ends. elapsed=24.76 s, mem usage=1.29 GB
[2026-08-27 04:33:34 TP3] Init torch distributed ends. elapsed=25.21 s, mem usage=1.29 GB
[2026-08-27 04:33:34 TP2] Init torch distributed ends. elapsed=25.75 s, mem usage=1.29 GB
[2026-08-27 04:33:34 TP1] Init torch distributed ends. elapsed=24.77 s, mem usage=1.29 GB
[2026-08-27 04:33:36 TP3] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP0] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP6] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP7] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP4] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP5] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP1] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP2] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 04:33:36 TP3] Load weight begin. avail mem=77.33 GB
[2026-08-27 04:33:36 TP6] Load weight begin. avail mem=77.33 GB
[2026-08-27 04:33:36 TP4] Load weight begin. avail mem=77.33 GB
[2026-08-27 04:33:36 TP0] Load weight begin. avail mem=77.38 GB
[2026-08-27 04:33:36 TP1] Load weight begin. avail mem=77.33 GB
[2026-08-27 04:33:36 TP7] Load weight begin. avail mem=77.57 GB
[2026-08-27 04:33:36 TP0] Detected fp8 checkpoint.
[2026-08-27 04:33:36 TP5] Load weight begin. avail mem=77.33 GB
[2026-08-27 04:33:36 TP2] Load weight begin. avail mem=77.33 GB
[2026-08-27 04:33:36 TP3] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:36 TP7] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:36 TP6] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:36 TP0] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:36 TP4] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:36 TP1] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:36 TP5] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:36 TP2] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 04:33:53 TP0] Shared experts fusion optimization enabled.
[FP8SafeTensorLoader] Detected format: deepseek
[FP8SafeTensorLoader] Detected scale format: block-wise (weight_scale_inv)

⏎ Multi-thread loading shards:   0% Completed | 0/163 [00:00<?, ?it/s]
⏎ Multi-thread loading shards:   1% Completed | 1/163 [00:00<01:42,  1.58it/s]
⏎ Multi-thread loading shards:   2% Completed | 3/163 [00:00<00:36,  4.40it/s]
⏎ Multi-thread loading shards:   3% Completed | 5/163 [00:00<00:23,  6.81it/s]
⏎ Multi-thread loading shards:   4% Completed | 7/163 [00:01<00:17,  8.94it/s]
⏎ Multi-thread loading shards:   6% Completed | 9/163 [00:01<00:14, 10.46it/s]
⏎ Multi-thread loading shards:   7% Completed | 11/163 [00:01<00:12, 11.87it/s]
⏎ Multi-thread loading shards:   8% Completed | 13/163 [00:01<00:10, 13.64it/s]
⏎ Multi-thread loading shards:   9% Completed | 15/163 [00:01<00:10, 14.13it/s]
⏎ Multi-thread loading shards:  10% Completed | 17/163 [00:01<00:09, 14.77it/s]
⏎ Multi-thread loading shards:  12% Completed | 19/163 [00:01<00:09, 15.01it/s]
⏎ Multi-thread loading shards:  13% Completed | 21/163 [00:01<00:09, 14.89it/s]
⏎ Multi-thread loading shards:  14% Completed | 23/163 [00:02<00:09, 14.72it/s]
⏎ Multi-thread loading shards:  15% Completed | 25/163 [00:02<00:08, 15.50it/s]
⏎ Multi-thread loading shards:  17% Completed | 27/163 [00:02<00:08, 15.69it/s]
⏎ Multi-thread loading shards:  18% Completed | 29/163 [00:02<00:08, 15.68it/s]
⏎ Multi-thread loading shards:  19% Completed | 31/163 [00:02<00:15,  8.34it/s]
⏎ Multi-thread loading shards:  21% Completed | 35/163 [00:03<00:10, 12.42it/s]
⏎ Multi-thread loading shards:  23% Completed | 38/163 [00:03<00:13,  9.40it/s]
⏎ Multi-thread loading shards:  25% Completed | 40/163 [00:03<00:12,  9.81it/s]
⏎ Multi-thread loading shards:  26% Completed | 42/163 [00:03<00:12,  9.88it/s]
⏎ Multi-thread loading shards:  27% Completed | 44/163 [00:04<00:11, 10.47it/s]
⏎ Multi-thread loading shards:  28% Completed | 46/163 [00:04<00:10, 11.47it/s]
⏎ Multi-thread loading shards:  29% Completed | 48/163 [00:04<00:09, 12.53it/s]
⏎ Multi-thread loading shards:  31% Completed | 50/163 [00:04<00:08, 13.28it/s]
⏎ Multi-thread loading shards:  32% Completed | 52/163 [00:04<00:07, 14.02it/s]
⏎ Multi-thread loading shards:  33% Completed | 54/163 [00:04<00:07, 14.58it/s]
⏎ Multi-thread loading shards:  35% Completed | 57/163 [00:04<00:06, 17.55it/s]
⏎ Multi-thread loading shards:  37% Completed | 60/163 [00:04<00:05, 19.50it/s]
⏎ Multi-thread loading shards:  39% Completed | 63/163 [00:05<00:10,  9.99it/s]
⏎ Multi-thread loading shards:  40% Completed | 65/163 [00:06<00:12,  7.65it/s]
⏎ Multi-thread loading shards:  41% Completed | 67/163 [00:06<00:11,  8.50it/s]
⏎ Multi-thread loading shards:  42% Completed | 69/163 [00:06<00:09,  9.88it/s]
⏎ Multi-thread loading shards:  44% Completed | 71/163 [00:06<00:08, 11.14it/s]
⏎ Multi-thread loading shards:  45% Completed | 73/163 [00:06<00:07, 12.16it/s]
⏎ Multi-thread loading shards:  46% Completed | 75/163 [00:06<00:06, 12.97it/s]
⏎ Multi-thread loading shards:  47% Completed | 77/163 [00:06<00:06, 13.83it/s]
⏎ Multi-thread loading shards:  49% Completed | 80/163 [00:06<00:05, 15.48it/s]
⏎ Multi-thread loading shards:  50% Completed | 82/163 [00:07<00:05, 15.69it/s]
⏎ Multi-thread loading shards:  52% Completed | 84/163 [00:07<00:05, 15.66it/s]
⏎ Multi-thread loading shards:  53% Completed | 86/163 [00:07<00:04, 15.40it/s]
⏎ Multi-thread loading shards:  54% Completed | 88/163 [00:07<00:04, 15.76it/s]
⏎ Multi-thread loading shards:  55% Completed | 90/163 [00:07<00:04, 15.86it/s]
⏎ Multi-thread loading shards:  56% Completed | 92/163 [00:07<00:04, 16.11it/s]
⏎ Multi-thread loading shards:  58% Completed | 94/163 [00:07<00:04, 15.85it/s]
⏎ Multi-thread loading shards:  59% Completed | 96/163 [00:07<00:04, 16.01it/s]
⏎ Multi-thread loading shards:  60% Completed | 98/163 [00:08<00:04, 15.86it/s]
⏎ Multi-thread loading shards:  61% Completed | 100/163 [00:08<00:03, 16.77it/s][2026-08-27 04:34:02 TP4] Load weight end. elapsed=25.80 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.
[2026-08-27 04:34:02 TP6] Load weight end. elapsed=25.90 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.
[2026-08-27 04:34:02 TP5] Load weight end. elapsed=26.04 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.
[2026-08-27 04:34:02 TP7] Load weight end. elapsed=26.08 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.34 GB, mem usage=3.23 GB.

⏎ Multi-thread loading shards:  63% Completed | 102/163 [00:08<00:08,  7.40it/s]
⏎ Multi-thread loading shards:  64% Completed | 104/163 [00:08<00:06,  8.81it/s]
⏎ Multi-thread loading shards:  65% Completed | 106/163 [00:09<00:05, 10.05it/s]
⏎ Multi-thread loading shards:  66% Completed | 108/163 [00:09<00:04, 11.17it/s]
⏎ Multi-thread loading shards:  67% Completed | 110/163 [00:09<00:04, 12.44it/s][2026-08-27 04:34:03 TP2] Load weight end. elapsed=26.79 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.
[2026-08-27 04:34:03 TP3] Load weight end. elapsed=26.83 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.

⏎ Multi-thread loading shards:  69% Completed | 112/163 [00:09<00:03, 13.34it/s]
⏎ Multi-thread loading shards:  70% Completed | 114/163 [00:09<00:03, 14.22it/s]
⏎ Multi-thread loading shards:  71% Completed | 116/163 [00:09<00:03, 14.46it/s][2026-08-27 04:34:03 TP1] Load weight end. elapsed=27.20 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.

⏎ Multi-thread loading shards:  73% Completed | 119/163 [00:09<00:02, 16.76it/s]
⏎ Multi-thread loading shards:  75% Completed | 123/163 [00:09<00:01, 21.08it/s]
⏎ Multi-thread loading shards:  77% Completed | 126/163 [00:10<00:01, 21.89it/s]
⏎ Multi-thread loading shards:  79% Completed | 129/163 [00:10<00:03, 11.14it/s]
⏎ Multi-thread loading shards:  80% Completed | 131/163 [00:10<00:02, 11.92it/s]
⏎ Multi-thread loading shards:  82% Completed | 133/163 [00:10<00:02, 12.69it/s]
⏎ Multi-thread loading shards:  83% Completed | 135/163 [00:11<00:02, 13.48it/s]
⏎ Multi-thread loading shards:  84% Completed | 137/163 [00:11<00:01, 14.27it/s]
⏎ Multi-thread loading shards:  85% Completed | 139/163 [00:11<00:01, 14.65it/s]
⏎ Multi-thread loading shards:  87% Completed | 141/163 [00:11<00:01, 15.23it/s]
⏎ Multi-thread loading shards:  88% Completed | 143/163 [00:11<00:01, 15.78it/s]
⏎ Multi-thread loading shards:  89% Completed | 145/163 [00:11<00:01, 15.90it/s]
⏎ Multi-thread loading shards:  90% Completed | 147/163 [00:12<00:02,  7.32it/s]
⏎ Multi-thread loading shards:  91% Completed | 149/163 [00:12<00:01,  8.64it/s]
⏎ Multi-thread loading shards:  93% Completed | 151/163 [00:12<00:01, 10.18it/s]
⏎ Multi-thread loading shards:  94% Completed | 153/163 [00:12<00:00, 11.47it/s]
⏎ Multi-thread loading shards:  95% Completed | 155/163 [00:12<00:00, 12.82it/s]
⏎ Multi-thread loading shards:  97% Completed | 158/163 [00:12<00:00, 15.41it/s]
⏎ Multi-thread loading shards: 100% Completed | 163/163 [00:12<00:00, 12.58it/s]
[2026-08-27 04:34:07 TP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 5048, in run_scheduler_process
    scheduler = Scheduler(
                ^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 518, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 995, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 921, in init_tp_model_worker
    self.tp_worker = TpModelWorker(**worker_kwargs)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 337, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 466, in _init_model_runner
    self._model_runner = ModelRunner(
                         ^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 436, in __init__
    self.initialize()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 630, in initialize
    self.load_model()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 1099, in load_model
    loaded = load_model_with_memory_saver(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py", line 321, in load_model_with_memory_saver
    model = loader.load_model(
            ^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py", line 991, in load_model
    self.load_weights_and_postprocess(
  File "/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py", line 1064, in load_weights_and_postprocess
    quant_method.process_weights_after_loading(module)
  File "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py", line 259, in process_weights_after_loading
    self.wrapper.load_weights(physical_to_logical_map_cpu)
  File "/usr/local/lib/python3.12/dist-packages/kt_kernel/utils/amx.py", line 944, in load_weights
    self.cpu_infer.sync()
ValueError: native FP8 per-expert TP source is incomplete

CPUInfer[0x1b544d30]: Hello
WorkerPool[0x1b268250] 2 subpools, [numa:threads][0:48] [1:48] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 48 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 48 threads
TP MOE layer 3, pool: 0x1b268250, expert num: 257, num_experts_per_tok: 9
Created FP8_MOE_TP 0 at numa 0 (backend=AMX)
Created FP8_MOE_TP 1 at numa 1 (backend=AMX)
[2026-08-27 04:34:07] Received sigquit from a child process. It usually means the child failed.
[2026-08-27 04:34:07] kill_process_tree called: parent_pid=9924, include_parent=True, pid=9924
```

# 원시 파일 — `eval/results/20260827_140008_tsk043_main_r1/`

R1-0528 본판 최종 — r1 AMXINT4 하이브리드 서빙

## 24. `eval/results/20260827_140008_tsk043_main_r1/RESULTS.md`

- 바이트 **2,425** · 줄 **40** · SHA256 `96ef2b9e05d0954643ee939aebf1a0d279a1e5eb79a1062face1643510cf98e0`
- 인코딩 utf-8

```text
# TSK_043 본판 — DeepSeek-R1-0528 KT Hybrid 측정 결과 (2026-08-27)

- 노드: violet-h100-016 (Xeon 8480+×2 AMX / 2TB DDR5 / H100×8, **turbo OFF 2.0GHz**)
- 모델: deepseek-ai/DeepSeek-R1-0528 (native FP8, **642GB** — HBM 총량 640GB 초과)
- 스택: SGLang 0.5.18 (lmsysorg/sglang:latest) + kt-kernel 0.7.0.post2 (--no-deps) + 호환성 패치 4건
- CPU expert: AMXINT4 변환본 (328GB, `kt quant -m int4 -i fp8`, 변환 65분) — 257 experts/layer 전량 CPU

## r0 — GPU-only 불가 실증 (TST_021 게이트 1) ✅

vLLM 0.28, TP=8, gmu 0.95 → 90초 만에 `torch.OutOfMemoryError: CUDA out of memory` (worker 로드 중).
**642GB > usable HBM ~608GB — 이 머신에서 hybrid 가 "선택"이 아니라 "필수"인 regime 실증.**

## r1 — KT Hybrid 서빙 (GPU MLA attention TP=8 + CPU AMX experts)

| 항목 | 값 |
|---|---|
| 기동 | HEALTH OK **160s** (642GB expert → DRAM, page cache 덕) |
| 처리량 | **19.67 out tok/s** (32 req × 1024/128, C=8, 전 req 완료) |
| TTFT p50 / TPOT p50 | 11,199 ms / 305.5 ms |
| CPU busy | **avg 42.7% / max 49.5%** (cpuinfer 96 threads 포화 — CLAUDE.md "CPU 활용" 직접 달성) |
| GPU util | avg 75.6% (MLA attention + 통신) |

### ⚠ 품질 게이트 미통과 (후속 SUB_167)

greedy smoke 출력이 비문 (따옴표/기호 스팸). **판별 실험 완료**:

| 실험 | 결과 |
|---|---|
| chat template 경유 | 동일 깨짐 → template 아님 |
| Qwen3-30B-A3B-FP8 → AMXINT8 | 출력 정상 |
| **Qwen3-30B-A3B-FP8 → AMXINT4 (동일 변환기)** | **출력 정상** → INT4 경로 일반 결함 아님 |
| 변환 스크립트 main vs v0.7.0.post1 태그 | IDENTICAL → 버전 스큐 아님 |
| kt-kernel 0.7.0 스왑 | 무관한 import 오류로 부팅 불가 — 판정 불가 |

→ **원인 후보를 "DeepSeek 계열 특이 처리" (block-wise `weight_scale_inv` 128×128 dequant 또는 shared expert(257번째) 폴딩) 로 좁힘**. upstream (kvcache-ai/ktransformers) 이슈 조회/제보 대상.

## 해석

- 처리량 19.7 tok/s 는 turbo OFF (2.0GHz) + expert 전량 CPU (`kt-num-gpu-experts 0`) + deferral 미사용의 **하한**. KT 공식 참조치 (8×L20+Xeon, R1 227 tok/s total) 대비 개선 여지: turbo unlock, hot expert GPU 배치 (`--kt-num-gpu-experts` 수십~수백), expert deferral, cpuinfer 튜닝
- 그러나 성능 논의는 **품질 게이트 통과 후**에만 유효 (SUB_167 선행)
```

## 25. `eval/results/20260827_140008_tsk043_main_r1/RUN.log`

- 바이트 **217** · 줄 **4** · SHA256 `79fcbb643ad6a9c6ef623346f389b0cc926be3ce30680751d78d70708c1ff567`
- 인코딩 utf-8

```text
== TSK_043 main start 20260827_140008 model=/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 ==
-- r1_kt_hybrid --
SGL HEALTH OK 160s
== TSK_043 main done 140700 ==
```

## 26. `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/.cpu_mon`

- 바이트 **6** · 줄 **1** · SHA256 `777413895674776b131b0e13c3bf67ba6e72a25740f831472569a1849523fb06`
- 인코딩 utf-8

```text
74883
```

## 27. `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/.gpu_mon`

- 바이트 **6** · 줄 **1** · SHA256 `b3845e7e695d8abd9d8d54bb6e2f72786e95b2d76bf180396170625c7ca80c26`
- 인코딩 utf-8

```text
74881
```

## 28. `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/bench.log`

- 바이트 **5,062** · 줄 **44** · SHA256 `e26d88ae1e792f30ef7e9ece150429151e5ac596b90a6bced13a022a0feb7c50`
- 인코딩 utf-8 · CR 6건 치환

```text
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f60cabe3240>, trust_remote_code=False, seed=42, num_prompts=32, dataset_name='sonnet', no_stream=False, dataset_path='/repo/benchmarks/sonnet.txt', no_oversample=False, skip_chat_template=False, enable_multimodal_chat=False, disable_shuffle=False, custom_output_len=256, custom_ensure_client_side_data=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=1024, sonnet_output_len=128, sonnet_prefix_len=200, sharegpt_output_len=None, timed_trace_chunk_hash_size=16, timed_trace_sec_multiplier=1, timed_trace_label_timestamp='timestamp', timed_trace_label_input_length='input_length', timed_trace_label_output_length='output_length', timed_trace_label_hash_ids='hash_ids', blazedit_min_distance=0.0, blazedit_max_distance=1.0, asr_max_audio_len_sec=inf, asr_min_audio_len_sec=0.0, random_input_len=1024, random_output_len=128, random_range_ratio='0.0', random_prefix_len=0, random_batch_size=1, no_reranker=False, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 1}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, bfcl_categories=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, speed_bench_dataset_subset='qualitative', speed_bench_output_len=4096, speed_bench_category=None, label=None, backend='openai', base_url='http://127.0.0.1:30000', host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=8, model='r1', input_len=None, output_len=None, tokenizer='/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52', tokenizer_mode='auto', use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, probe_request_rate=0.0, disable_tqdm=False, num_warmups=0, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, self_timed=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='50,95', goodput=None, request_id_prefix='bench-300e875e-', top_p=None, top_k=None, min_p=None, temperature=None, frequency_penalty=None, presence_penalty=None, repetition_penalty=None, served_model_name=None, lora_modules=None, lora_assignment='random', ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=0, chat_template_kwargs=None, extra_body=None, skip_tokenizer_init=False, insecure=False, plot_timeline=False, timeline_itl_thresholds='25,50', plot_dataset_stats=False)
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 8

⏎   0%|          | 0/32 [00:00<?, ?it/s]
⏎   3%|▎         | 1/32 [00:56<29:19, 56.76s/it]
⏎  28%|██▊       | 9/32 [01:47<03:58, 10.35s/it]
⏎  53%|█████▎    | 17/32 [02:38<02:01,  8.11s/it]
⏎  78%|███████▊  | 25/32 [03:28<00:50,  7.27s/it]
⏎ 100%|██████████| 32/32 [03:28<00:00,  6.51s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  208.26    
Total input tokens:                      32510     
Total generated tokens:                  4096      
Request throughput (req/s):              0.15      
Output token throughput (tok/s):         19.67     
Peak output token throughput (tok/s):    32.00     
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          175.77    
---------------Time to First Token----------------
Mean TTFT (ms):                          12380.89  
Median TTFT (ms):                        11199.23  
P50 TTFT (ms):                           11199.23  
P95 TTFT (ms):                           18098.75  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          312.46    
Median TPOT (ms):                        305.50    
P50 TPOT (ms):                           305.50    
P95 TPOT (ms):                           326.34    
---------------Inter-token Latency----------------
Mean ITL (ms):                           312.46    
Median ITL (ms):                         304.82    
P50 ITL (ms):                            304.82    
P95 ITL (ms):                            318.48    
==================================================
```

## 29. `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/cpu_util.txt`

- 바이트 **2,587** · 줄 **118** · SHA256 `7384b2ee3997011c428d48e293b57b93054c26388bdd7a2cfe2e2aa4a280df53`
- 인코딩 utf-8

```text
1787806979 busy=4.69
1787806981 busy=16.93
1787806983 busy=47.38
1787806985 busy=46.66
1787806987 busy=46.37
1787806989 busy=46.51
1787806991 busy=27.76
1787806993 busy=3.48
1787806995 busy=3.23
1787806997 busy=3.29
1787806999 busy=3.30
1787807001 busy=7.09
1787807003 busy=3.70
1787807005 busy=3.24
1787807007 busy=15.70
1787807009 busy=47.54
1787807011 busy=4.76
1787807013 busy=24.70
1787807015 busy=48.11
1787807017 busy=47.69
1787807019 busy=47.44
1787807021 busy=47.49
1787807023 busy=47.55
1787807025 busy=40.71
1787807027 busy=46.87
1787807029 busy=46.92
1787807031 busy=46.76
1787807033 busy=46.69
1787807035 busy=46.87
1787807037 busy=46.76
1787807039 busy=47.19
1787807041 busy=46.89
1787807043 busy=47.35
1787807045 busy=47.21
1787807047 busy=46.83
1787807049 busy=46.92
1787807051 busy=46.79
1787807053 busy=47.96
1787807055 busy=46.76
1787807057 busy=46.78
1787807059 busy=46.57
1787807061 busy=47.60
1787807063 busy=34.36
1787807065 busy=47.24
1787807067 busy=47.26
1787807069 busy=45.45
1787807071 busy=47.67
1787807073 busy=47.56
1787807075 busy=47.53
1787807077 busy=47.06
1787807079 busy=46.92
1787807081 busy=46.78
1787807083 busy=47.02
1787807085 busy=46.73
1787807087 busy=47.18
1787807089 busy=47.26
1787807091 busy=47.02
1787807093 busy=46.85
1787807095 busy=46.87
1787807097 busy=47.26
1787807099 busy=47.11
1787807101 busy=46.65
1787807103 busy=47.46
1787807105 busy=46.89
1787807107 busy=47.77
1787807109 busy=46.85
1787807111 busy=46.73
1787807113 busy=46.74
1787807115 busy=47.30
1787807117 busy=47.16
1787807119 busy=47.47
1787807121 busy=46.78
1787807123 busy=47.95
1787807125 busy=46.94
1787807127 busy=47.96
1787807129 busy=47.30
1787807131 busy=47.23
1787807133 busy=47.34
1787807135 busy=47.43
1787807137 busy=46.83
1787807140 busy=46.66
1787807142 busy=46.87
1787807144 busy=46.90
1787807146 busy=46.60
1787807148 busy=46.83
1787807150 busy=46.67
1787807152 busy=46.62
1787807154 busy=47.13
1787807156 busy=47.06
1787807158 busy=46.91
1787807160 busy=46.77
1787807162 busy=46.92
1787807164 busy=47.32
1787807166 busy=47.21
1787807168 busy=47.15
1787807170 busy=48.14
1787807172 busy=47.06
1787807174 busy=47.57
1787807176 busy=47.09
1787807178 busy=46.86
1787807180 busy=46.66
1787807182 busy=46.98
1787807184 busy=47.32
1787807186 busy=47.02
1787807188 busy=49.49
1787807190 busy=46.70
1787807192 busy=47.06
1787807194 busy=47.41
1787807196 busy=47.07
1787807198 busy=46.80
1787807200 busy=46.80
1787807202 busy=46.77
1787807204 busy=46.98
1787807206 busy=46.78
1787807208 busy=47.52
1787807210 busy=47.33
1787807212 busy=46.76
1787807214 busy=43.33
```

## 30. `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/gpu_util.csv`

- 바이트 **11,306** · 줄 **384** · SHA256 `159b137b0d8d62e01002aa3bd0c6ee99fcc94d7d24397fe44c901272bb177e48`
- 인코딩 utf-8

```text
0, 0 %, 72464 MiB, 123.00 W
1, 0 %, 72560 MiB, 122.10 W
2, 0 %, 72560 MiB, 122.86 W
3, 0 %, 72560 MiB, 127.20 W
4, 1 %, 72560 MiB, 125.29 W
5, 0 %, 72560 MiB, 118.83 W
6, 0 %, 72560 MiB, 126.09 W
7, 1 %, 72080 MiB, 126.54 W
0, 7 %, 72464 MiB, 147.65 W
1, 97 %, 72560 MiB, 152.04 W
2, 59 %, 72560 MiB, 153.64 W
3, 100 %, 72560 MiB, 160.73 W
4, 100 %, 72560 MiB, 158.77 W
5, 94 %, 72560 MiB, 151.11 W
6, 100 %, 72560 MiB, 157.41 W
7, 100 %, 72080 MiB, 159.72 W
0, 5 %, 72464 MiB, 147.93 W
1, 76 %, 72560 MiB, 152.60 W
2, 100 %, 72560 MiB, 154.21 W
3, 100 %, 72560 MiB, 160.93 W
4, 100 %, 72560 MiB, 159.04 W
5, 95 %, 72560 MiB, 151.24 W
6, 98 %, 72560 MiB, 157.49 W
7, 100 %, 72080 MiB, 159.67 W
0, 0 %, 72464 MiB, 116.36 W
1, 0 %, 72560 MiB, 113.95 W
2, 0 %, 72560 MiB, 115.54 W
3, 0 %, 72560 MiB, 121.24 W
4, 0 %, 72560 MiB, 119.10 W
5, 0 %, 72560 MiB, 112.54 W
6, 0 %, 72560 MiB, 117.47 W
7, 0 %, 72080 MiB, 119.24 W
0, 0 %, 72464 MiB, 116.40 W
1, 0 %, 72560 MiB, 113.82 W
2, 0 %, 72560 MiB, 115.47 W
3, 0 %, 72560 MiB, 121.18 W
4, 0 %, 72560 MiB, 119.15 W
5, 0 %, 72560 MiB, 112.51 W
6, 0 %, 72560 MiB, 117.48 W
7, 0 %, 72080 MiB, 119.21 W
0, 0 %, 72464 MiB, 116.42 W
1, 0 %, 72560 MiB, 113.84 W
2, 0 %, 72560 MiB, 115.41 W
3, 0 %, 72560 MiB, 121.18 W
4, 0 %, 72560 MiB, 118.46 W
5, 0 %, 72560 MiB, 112.48 W
6, 0 %, 72560 MiB, 117.48 W
7, 0 %, 72080 MiB, 119.62 W
0, 3 %, 72794 MiB, 133.20 W
1, 100 %, 72764 MiB, 134.45 W
2, 100 %, 72764 MiB, 136.46 W
3, 100 %, 72764 MiB, 145.20 W
4, 100 %, 72764 MiB, 141.12 W
5, 100 %, 72764 MiB, 135.41 W
6, 100 %, 72764 MiB, 138.12 W
7, 100 %, 72284 MiB, 141.27 W
0, 43 %, 73368 MiB, 127.76 W
1, 0 %, 74342 MiB, 125.80 W
2, 97 %, 74342 MiB, 126.51 W
3, 100 %, 74342 MiB, 135.07 W
4, 84 %, 74342 MiB, 132.69 W
5, 0 %, 74342 MiB, 124.67 W
6, 0 %, 74342 MiB, 129.27 W
7, 100 %, 73862 MiB, 132.04 W
0, 10 %, 75124 MiB, 136.37 W
1, 100 %, 74342 MiB, 135.03 W
2, 100 %, 74342 MiB, 134.63 W
3, 100 %, 74342 MiB, 142.00 W
4, 100 %, 74342 MiB, 138.92 W
5, 100 %, 74342 MiB, 134.36 W
6, 100 %, 74342 MiB, 141.29 W
7, 100 %, 73862 MiB, 142.29 W
0, 0 %, 75124 MiB, 135.03 W
1, 100 %, 74342 MiB, 133.91 W
2, 100 %, 74342 MiB, 138.16 W
3, 100 %, 74342 MiB, 144.39 W
4, 100 %, 74342 MiB, 144.24 W
5, 100 %, 74342 MiB, 137.58 W
6, 100 %, 74342 MiB, 139.26 W
7, 100 %, 73862 MiB, 144.44 W
0, 19 %, 75124 MiB, 164.70 W
1, 100 %, 74342 MiB, 198.69 W
2, 30 %, 74342 MiB, 160.82 W
3, 100 %, 74342 MiB, 212.52 W
4, 100 %, 74342 MiB, 209.20 W
5, 100 %, 74342 MiB, 201.08 W
6, 100 %, 74342 MiB, 206.54 W
7, 100 %, 73862 MiB, 209.28 W
0, 27 %, 75124 MiB, 164.30 W
1, 100 %, 74342 MiB, 201.56 W
2, 29 %, 74342 MiB, 160.84 W
3, 100 %, 74342 MiB, 213.01 W
4, 100 %, 74342 MiB, 210.43 W
5, 100 %, 74342 MiB, 201.43 W
6, 93 %, 74342 MiB, 206.75 W
7, 100 %, 73862 MiB, 210.32 W
0, 30 %, 75124 MiB, 167.90 W
1, 100 %, 74342 MiB, 202.77 W
2, 27 %, 74342 MiB, 159.30 W
3, 100 %, 74342 MiB, 212.62 W
4, 100 %, 74342 MiB, 210.87 W
5, 100 %, 74342 MiB, 201.66 W
6, 93 %, 74342 MiB, 207.94 W
7, 100 %, 73862 MiB, 211.08 W
0, 28 %, 75124 MiB, 167.01 W
1, 100 %, 74342 MiB, 202.75 W
2, 22 %, 74342 MiB, 159.97 W
3, 100 %, 74342 MiB, 213.72 W
4, 100 %, 74342 MiB, 212.38 W
5, 100 %, 74342 MiB, 202.15 W
6, 100 %, 74342 MiB, 207.49 W
7, 100 %, 73862 MiB, 211.12 W
0, 28 %, 75124 MiB, 167.70 W
1, 100 %, 74342 MiB, 202.83 W
2, 31 %, 74342 MiB, 159.95 W
3, 100 %, 74342 MiB, 213.39 W
4, 100 %, 74342 MiB, 211.74 W
5, 100 %, 74342 MiB, 202.12 W
6, 93 %, 74342 MiB, 208.34 W
7, 100 %, 73862 MiB, 211.43 W
0, 25 %, 75124 MiB, 166.59 W
1, 100 %, 74342 MiB, 202.78 W
2, 26 %, 74342 MiB, 160.87 W
3, 100 %, 74342 MiB, 213.82 W
4, 100 %, 74342 MiB, 211.59 W
5, 100 %, 74342 MiB, 200.99 W
6, 100 %, 74342 MiB, 206.85 W
7, 100 %, 73862 MiB, 211.24 W
0, 8 %, 75124 MiB, 150.91 W
1, 100 %, 74342 MiB, 195.74 W
2, 93 %, 74342 MiB, 194.84 W
3, 100 %, 74342 MiB, 206.44 W
4, 99 %, 74342 MiB, 207.84 W
5, 100 %, 74342 MiB, 196.98 W
6, 94 %, 74342 MiB, 204.32 W
7, 100 %, 73862 MiB, 207.65 W
0, 29 %, 75124 MiB, 166.27 W
1, 100 %, 74342 MiB, 202.61 W
2, 31 %, 74342 MiB, 164.08 W
3, 100 %, 74342 MiB, 214.24 W
4, 100 %, 74342 MiB, 212.40 W
5, 100 %, 74342 MiB, 201.30 W
6, 93 %, 74342 MiB, 206.35 W
7, 100 %, 73862 MiB, 210.48 W
0, 5 %, 75124 MiB, 142.87 W
1, 100 %, 74342 MiB, 143.93 W
2, 100 %, 74342 MiB, 143.03 W
3, 100 %, 74342 MiB, 152.62 W
4, 100 %, 74342 MiB, 152.23 W
5, 100 %, 74342 MiB, 143.42 W
6, 100 %, 74342 MiB, 148.82 W
7, 100 %, 73862 MiB, 150.75 W
0, 5 %, 75124 MiB, 145.43 W
1, 100 %, 74342 MiB, 145.27 W
2, 100 %, 74342 MiB, 146.04 W
3, 100 %, 74342 MiB, 151.22 W
4, 100 %, 74342 MiB, 150.38 W
5, 100 %, 74342 MiB, 142.02 W
6, 100 %, 74342 MiB, 149.35 W
7, 100 %, 73862 MiB, 152.60 W
0, 29 %, 75124 MiB, 167.56 W
1, 100 %, 74342 MiB, 203.26 W
2, 29 %, 74342 MiB, 162.21 W
3, 100 %, 74342 MiB, 214.27 W
4, 100 %, 74342 MiB, 210.80 W
5, 100 %, 74342 MiB, 200.55 W
6, 100 %, 74342 MiB, 207.58 W
7, 100 %, 73862 MiB, 212.02 W
0, 27 %, 75124 MiB, 165.59 W
1, 100 %, 74342 MiB, 203.07 W
2, 31 %, 74342 MiB, 162.46 W
3, 100 %, 74342 MiB, 215.18 W
4, 100 %, 74342 MiB, 211.34 W
5, 100 %, 74342 MiB, 201.41 W
6, 93 %, 74342 MiB, 207.04 W
7, 100 %, 73862 MiB, 211.68 W
0, 27 %, 75124 MiB, 164.88 W
1, 100 %, 74342 MiB, 203.42 W
2, 27 %, 74342 MiB, 163.46 W
3, 100 %, 74342 MiB, 214.11 W
4, 100 %, 74342 MiB, 211.78 W
5, 100 %, 74342 MiB, 200.83 W
6, 100 %, 74342 MiB, 206.48 W
7, 100 %, 73862 MiB, 212.55 W
0, 29 %, 75124 MiB, 167.16 W
1, 100 %, 74342 MiB, 203.67 W
2, 28 %, 74342 MiB, 162.75 W
3, 100 %, 74342 MiB, 215.47 W
4, 100 %, 74342 MiB, 211.71 W
5, 100 %, 74342 MiB, 201.68 W
6, 93 %, 74342 MiB, 206.61 W
7, 100 %, 73862 MiB, 213.12 W
0, 27 %, 75124 MiB, 169.85 W
1, 100 %, 74342 MiB, 204.78 W
2, 31 %, 74342 MiB, 181.13 W
3, 100 %, 74342 MiB, 216.88 W
4, 100 %, 74342 MiB, 213.68 W
5, 100 %, 74342 MiB, 202.97 W
6, 74 %, 74342 MiB, 188.32 W
7, 100 %, 73862 MiB, 214.30 W
0, 29 %, 75124 MiB, 169.80 W
1, 100 %, 74342 MiB, 203.61 W
2, 26 %, 74342 MiB, 160.76 W
3, 100 %, 74342 MiB, 216.17 W
4, 100 %, 74342 MiB, 211.38 W
5, 100 %, 74342 MiB, 202.48 W
6, 94 %, 74342 MiB, 209.11 W
7, 100 %, 73862 MiB, 213.79 W
0, 29 %, 75124 MiB, 166.35 W
1, 100 %, 74342 MiB, 202.95 W
2, 29 %, 74342 MiB, 163.82 W
3, 100 %, 74342 MiB, 215.95 W
4, 100 %, 74342 MiB, 212.45 W
5, 100 %, 74342 MiB, 201.85 W
6, 100 %, 74342 MiB, 207.92 W
7, 100 %, 73862 MiB, 214.04 W
0, 33 %, 75124 MiB, 167.62 W
1, 100 %, 74342 MiB, 203.49 W
2, 100 %, 74342 MiB, 170.14 W
3, 100 %, 74342 MiB, 216.48 W
4, 100 %, 74342 MiB, 213.58 W
5, 100 %, 74342 MiB, 202.12 W
6, 75 %, 74342 MiB, 201.64 W
7, 100 %, 73862 MiB, 214.65 W
0, 4 %, 75124 MiB, 141.22 W
1, 100 %, 74342 MiB, 140.74 W
2, 100 %, 74342 MiB, 141.55 W
3, 100 %, 74342 MiB, 150.37 W
4, 100 %, 74342 MiB, 149.79 W
5, 100 %, 74342 MiB, 138.94 W
6, 100 %, 74342 MiB, 145.16 W
7, 100 %, 73862 MiB, 148.67 W
0, 6 %, 75124 MiB, 154.64 W
1, 100 %, 74342 MiB, 158.73 W
2, 100 %, 74342 MiB, 158.95 W
3, 100 %, 74342 MiB, 169.90 W
4, 100 %, 74342 MiB, 167.95 W
5, 100 %, 74342 MiB, 156.79 W
6, 100 %, 74342 MiB, 163.51 W
7, 100 %, 73862 MiB, 166.44 W
0, 8 %, 75124 MiB, 167.89 W
1, 100 %, 74342 MiB, 203.26 W
2, 73 %, 74342 MiB, 164.98 W
3, 100 %, 74342 MiB, 216.24 W
4, 100 %, 74342 MiB, 214.74 W
5, 100 %, 74342 MiB, 201.65 W
6, 100 %, 74342 MiB, 208.60 W
7, 100 %, 73862 MiB, 214.12 W
0, 8 %, 75124 MiB, 153.59 W
1, 92 %, 74342 MiB, 195.06 W
2, 60 %, 74342 MiB, 194.11 W
3, 92 %, 74342 MiB, 211.70 W
4, 95 %, 74342 MiB, 210.64 W
5, 76 %, 74342 MiB, 193.57 W
6, 100 %, 74342 MiB, 204.85 W
7, 70 %, 73862 MiB, 205.07 W
0, 9 %, 75124 MiB, 153.58 W
1, 100 %, 74342 MiB, 194.46 W
2, 89 %, 74342 MiB, 192.97 W
3, 93 %, 74342 MiB, 211.87 W
4, 100 %, 74342 MiB, 210.58 W
5, 100 %, 74342 MiB, 192.91 W
6, 100 %, 74342 MiB, 205.71 W
7, 89 %, 73862 MiB, 204.76 W
0, 5 %, 75124 MiB, 154.45 W
1, 100 %, 74342 MiB, 196.46 W
2, 100 %, 74342 MiB, 193.96 W
3, 100 %, 74342 MiB, 213.70 W
4, 100 %, 74342 MiB, 211.20 W
5, 100 %, 74342 MiB, 193.77 W
6, 100 %, 74342 MiB, 206.46 W
7, 100 %, 73862 MiB, 206.77 W
0, 5 %, 75124 MiB, 154.12 W
1, 96 %, 74342 MiB, 195.71 W
2, 100 %, 74342 MiB, 194.67 W
3, 100 %, 74342 MiB, 213.83 W
4, 100 %, 74342 MiB, 211.55 W
5, 100 %, 74342 MiB, 193.19 W
6, 94 %, 74342 MiB, 206.10 W
7, 100 %, 73862 MiB, 205.77 W
0, 5 %, 75124 MiB, 153.72 W
1, 78 %, 74342 MiB, 195.56 W
2, 98 %, 74342 MiB, 194.61 W
3, 100 %, 74342 MiB, 213.12 W
4, 100 %, 74342 MiB, 211.86 W
5, 91 %, 74342 MiB, 194.55 W
6, 95 %, 74342 MiB, 205.76 W
7, 100 %, 73862 MiB, 206.18 W
0, 7 %, 75124 MiB, 153.99 W
1, 93 %, 74342 MiB, 194.61 W
2, 67 %, 74342 MiB, 194.40 W
3, 92 %, 74342 MiB, 212.84 W
4, 95 %, 74342 MiB, 212.04 W
5, 77 %, 74342 MiB, 194.23 W
6, 100 %, 74342 MiB, 205.62 W
7, 72 %, 73862 MiB, 205.72 W
0, 8 %, 75124 MiB, 154.52 W
1, 100 %, 74342 MiB, 194.09 W
2, 93 %, 74342 MiB, 193.63 W
3, 91 %, 74342 MiB, 212.94 W
4, 100 %, 74342 MiB, 213.52 W
5, 100 %, 74342 MiB, 193.50 W
6, 100 %, 74342 MiB, 205.59 W
7, 92 %, 73862 MiB, 204.71 W
0, 8 %, 75124 MiB, 149.96 W
1, 100 %, 74342 MiB, 147.61 W
2, 100 %, 74342 MiB, 148.93 W
3, 100 %, 74342 MiB, 160.50 W
4, 100 %, 74342 MiB, 161.18 W
5, 100 %, 74342 MiB, 144.60 W
6, 100 %, 74342 MiB, 155.34 W
7, 100 %, 73862 MiB, 155.73 W
0, 0 %, 75124 MiB, 141.97 W
1, 100 %, 74342 MiB, 138.67 W
2, 100 %, 74342 MiB, 140.18 W
3, 100 %, 74342 MiB, 153.09 W
4, 100 %, 74342 MiB, 152.89 W
5, 100 %, 74342 MiB, 140.52 W
6, 100 %, 74342 MiB, 143.40 W
7, 100 %, 73862 MiB, 147.34 W
0, 8 %, 75124 MiB, 156.28 W
1, 100 %, 74342 MiB, 194.93 W
2, 100 %, 74342 MiB, 195.34 W
3, 93 %, 74342 MiB, 213.29 W
4, 96 %, 74342 MiB, 214.24 W
5, 100 %, 74342 MiB, 193.41 W
6, 100 %, 74342 MiB, 205.65 W
7, 100 %, 73862 MiB, 204.37 W
0, 5 %, 75124 MiB, 156.08 W
1, 100 %, 74342 MiB, 195.87 W
2, 100 %, 74342 MiB, 195.19 W
3, 100 %, 74342 MiB, 212.80 W
4, 98 %, 74342 MiB, 214.42 W
5, 100 %, 74342 MiB, 193.57 W
6, 100 %, 74342 MiB, 205.72 W
7, 100 %, 73862 MiB, 206.02 W
0, 29 %, 75124 MiB, 164.91 W
1, 100 %, 74342 MiB, 200.51 W
2, 100 %, 74342 MiB, 189.28 W
3, 100 %, 74342 MiB, 214.59 W
4, 100 %, 74342 MiB, 215.14 W
5, 100 %, 74342 MiB, 198.71 W
6, 99 %, 74342 MiB, 196.59 W
7, 37 %, 73862 MiB, 206.21 W
0, 24 %, 75124 MiB, 170.74 W
1, 32 %, 74342 MiB, 163.52 W
2, 100 %, 74342 MiB, 207.56 W
3, 100 %, 74342 MiB, 218.30 W
4, 100 %, 74342 MiB, 220.29 W
5, 52 %, 74342 MiB, 180.56 W
6, 100 %, 74342 MiB, 211.50 W
7, 80 %, 73862 MiB, 203.45 W
0, 29 %, 75124 MiB, 170.38 W
1, 33 %, 74342 MiB, 164.16 W
2, 100 %, 74342 MiB, 208.26 W
3, 100 %, 74342 MiB, 218.01 W
4, 100 %, 74342 MiB, 220.70 W
5, 61 %, 74342 MiB, 181.88 W
6, 100 %, 74342 MiB, 211.85 W
7, 92 %, 73862 MiB, 202.19 W
0, 23 %, 75124 MiB, 169.98 W
1, 33 %, 74342 MiB, 165.39 W
2, 100 %, 74342 MiB, 207.73 W
3, 100 %, 74342 MiB, 218.51 W
4, 100 %, 74342 MiB, 220.60 W
5, 46 %, 74342 MiB, 182.84 W
6, 100 %, 74342 MiB, 211.34 W
7, 87 %, 73862 MiB, 204.80 W
0, 30 %, 75124 MiB, 171.75 W
1, 30 %, 74342 MiB, 165.95 W
2, 100 %, 74342 MiB, 208.48 W
3, 100 %, 74342 MiB, 219.10 W
4, 100 %, 74342 MiB, 220.75 W
5, 98 %, 74342 MiB, 181.60 W
6, 100 %, 74342 MiB, 211.96 W
7, 99 %, 73862 MiB, 202.30 W
0, 23 %, 75124 MiB, 170.66 W
1, 31 %, 74342 MiB, 164.12 W
2, 100 %, 74342 MiB, 208.16 W
3, 100 %, 74342 MiB, 218.62 W
4, 100 %, 74342 MiB, 220.29 W
5, 56 %, 74342 MiB, 183.21 W
6, 100 %, 74342 MiB, 211.43 W
7, 88 %, 73862 MiB, 204.34 W
```

## 31. `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/server.log`

- 바이트 **83,346** · 줄 **633** · SHA256 `71819cd9a350bd0929c13fe58883b09056b6205b200c5622d437dc6ab1b8bcc5`
- 인코딩 utf-8 · ANSI 시퀀스 2건 제거 · CR 77건 치환

```text
/sgl-workspace/sglang/python/sglang/launch_server.py:57: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
  Example: sglang serve --model-path <model> [options]
  warnings.warn(
'--disable-cuda-graph' is deprecated and will be removed in a future release. Use '--cuda-graph-backend-{decode,prefill}=disabled' instead.
[2026-08-27 05:00:23] Auto-enabling FlashInfer AllReduce Fusion on SM90/SM10X for DeepseekV3ForCausalLM
[2026-08-27 05:00:25] server_args=ServerArgs(model_path='/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52', tokenizer_path='/models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52', tokenizer_mode='auto', tokenizer_backend='huggingface', tokenizer_worker_num=1, detokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=True, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', model_config_parser='auto', json_model_override_args='{}', dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, enable_tf32_matmul=False, mem_fraction_static=0.831, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, retraction_policy='length', schedule_conservativeness=1.0, page_size=1, c128_page_size=16, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', prefill_only_disable_kv_cache=False, disable_radix_cache=False, enable_page_major_kv_layout=False, enable_unified_memory=False, disable_chunked_prefix_cache=False, disable_overlap_schedule=False, num_continuous_decode_steps=1, scheduler_recv_interval=1, enable_mixed_chunk=False, nccl_port=None, dist_timeout=None, dist_init_addr=None, nnodes=1, node_rank=0, tp_size=8, dcp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dwdp_size=1, dcp_comm_backend='ag_rs', dcp_replicate_q_proj=None, enable_prefill_cp=False, cp_strategy=None, enable_dsa_cache_layer_split=False, enable_dsa_prefill_context_parallel=False, dsa_prefill_cp_mode='round-robin-split', enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_cp_decode_attn_tp=False, enable_dp_attention=False, enable_dp_attention_local_control_broadcast=False, enable_dp_lm_head=False, enable_tp_lm_head_all_to_all=False, enable_attn_tp_input_scattered=False, disable_attn_tp_gather=False, enable_p2p_check=False, device='cuda', base_gpu_id=0, gpu_id_step=1, random_seed=239012767, mlx_enable_sampling=False, watchdog_timeout=300, soft_watchdog_timeout=None, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, numa_node=None, gc_threshold=None, host='127.0.0.1', port=30000, fastapi_root_path='', smg_grpc_mode=False, grpc_mode=False, grpc_port=None, sidecar=None, sidecar_args=None, skip_server_warmup=False, warmups=None, enable_http2=False, http2_max_concurrent_streams=200, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, api_key=None, admin_api_key=None, served_model_name='r1', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, default_chat_template_kwargs=None, strip_thinking_cache=False, enable_strict_thinking=False, tool_call_parser=None, tool_server=None, sampling_defaults='model', asr_max_buffer_seconds=60, asr_max_concurrent_sessions=32, preferred_sampling_params=None, allow_auto_truncate=False, stream_interval=1, batch_notify_size=16, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, enable_session_radix_cache=False, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, smg_http_sidecar_port=None, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, load_snapshot_publish_interval=15, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_forward_pass_metrics=False, forward_pass_metrics_worker_id='', forward_pass_metrics_ipc_name=None, enable_trace=False, trace_modules='request', otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, stat_loggers=None, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', radix_cache_backend=None, mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', bf16_gemm_backend='auto', dsa_prefill_backend=None, dsa_decode_backend=None, dsa_paged_mqa_logits_backend='auto', dsa_topk_backend='sgl-kernel', disable_flashinfer_autotune=False, flashinfer_autotune_skip_ops=None, mamba_backend='triton', cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend='disabled', max_bs=512, bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], tc_compiler='eager', full_prefill_max_req=None, full_prefill_prefix_chunk_tokens=None), prefill=PhaseConfig(backend='disabled', max_bs=2048, bs=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048], tc_compiler='eager', full_prefill_max_req=None, full_prefill_prefix_chunk_tokens=None)), cuda_graph_backend_decode=None, cuda_graph_backend_prefill=None, cuda_graph_max_bs_decode=None, cuda_graph_max_bs_prefill=None, cuda_graph_bs_decode=None, cuda_graph_bs_prefill=None, cuda_graph_tc_compiler=None, disable_prefill_cuda_graph=False, disable_decode_cuda_graph=False, disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, debug_cuda_graph=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, flashinfer_mla_disable_ragged=False, enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_deepseek_v4_fp4_indexer=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, enable_scattered_sconv=False, pre_warm_nccl=False, enable_quant_communications=False, enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, flashinfer_allreduce_fusion_backend='auto', enable_aiter_allreduce_fusion=False, enable_torch_compile=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_dflash_block_size=None, speculative_dspark_block_size=None, speculative_dspark_sps_table_path=None, speculative_dspark_confidence_sts_path=None, speculative_dspark_align_verify_tokens_to_graph_tier=False, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_use_rejection_sampling=False, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_draft_kv_cache_dtype=None, speculative_draft_window_size=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, _speculative_draft_quantization_explicitly_set=False, speculative_skip_dp_mlp_sync=False, enable_multi_layer_eagle=False, speculative_adaptive=False, speculative_adaptive_config=None, decoupled_spec_bind_endpoint=None, decoupled_spec_connect_endpoints=None, decoupled_spec_rank=None, decoupled_spec_role='null', spec_trace_dir=None, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, speculative_ngram_external_corpus_path=None, speculative_ngram_external_sam_budget=0, speculative_ngram_external_corpus_max_tokens=10000000, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', deepep_mode='auto', fuseep_mode=2, deepep_dispatcher_output_dtype='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, expert_balancedness_report_mode='off', deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, enable_waterfill=False, ep_join_mode=None, ep_join_rank_offset=0, elastic_ep_initial_size=None, max_ep_size=None, elastic_ep_scale_timeout=600, elastic_ep_rejoin=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, disable_shared_experts_fusion=False, enforce_shared_experts_fusion=False, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_max_states_per_path=-1, enable_mamba_cache_stochastic_rounding=False, mamba_cache_philox_rounds=0, mamba_full_memory_ratio=0.9, mamba_radix_cache_strategy='auto', uses_mamba_radix_cache=False, mamba_track_interval=256, enable_int8_mamba_checkpoint=False, int8_mamba_ckpt_size=None, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, linear_attn_verify_backend=None, enable_linear_replayssm=False, linear_replayssm_cache_len=16, enable_linear_replayssm_spec=False, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='page_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='timeout', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, mm_processor_worker_num=0, mm_io_worker_num=0, allowed_media_domains=[], media_url_max_file_size_mb=64, mm_preprocess_cache_size_mb=None, trust_mm_content_hashes=False, limit_mm_data_per_request=None, enable_mm_global_cache=False, image_processor_backend='auto', mm_global_cache_backend='mooncake', disable_fast_image_processor=False, mm_feature_transport='cpu', keep_mm_feature_on_device=False, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, lora_use_virtual_experts=False, lora_strict_loading=False, lora_drain_wait_threshold=0.0, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', enable_lmcache=False, lmcache_config_file=None, enable_flexkv=False, flexkv_config_file=None, kt_weight_path='/models/kt/r1-0528-int4', kt_method='AMXINT4', kt_cpuinfer=96, kt_threadpool_count=2, kt_num_gpu_experts=0, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, dllm_fdfo=True, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_radix_cache=False, disaggregation_decode_enable_offload_kvcache=False, disaggregation_decode_retraction_backup=None, num_reserved_decode_tokens=512, disaggregation_decode_extra_slots=None, disaggregation_decode_polling_interval=1, optimistic_prefill_attempts=0, encoder_only=False, language_only=False, language_model_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], encoder_bootstrap_port=8997, encoder_register_urls=[], enable_adaptive_dispatch_to_encoder=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, startup_weight_load_mode='serial', custom_weight_loader=[], weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False, weight_loader_prefetch_num_threads=4, weight_loader_drop_cache_after_load=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, download_dir=None, model_checksum=None, delete_ckpt_after_loading=False, decrypted_config_file=None, decrypted_draft_config_file=None, checkpoint_engine_wait_weights_before_ready=False, enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, prefill_delayer_queue_min_ratio=None, prefill_delayer_max_delay_ms=None, min_free_slots_delay=None, enable_deterministic_inference=False, rl_on_policy_target=None, kv_canary='none', kv_canary_real_data='none', kv_canary_sweep_interval=0, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, enable_custom_logit_processor=False, enable_return_hidden_states=False, return_hidden_states_mode=None, enable_return_routed_experts=False, enable_return_indexer_topk=False, disable_outlines_disk_cache=False, enable_mis=False, weight_cache_mode='off', weight_cache_socket=None, weight_cache_timeout=1800, forward_hooks=None, msprobe_dump_config=None)
[2026-08-27 05:00:32] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:32] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:32] Using default HuggingFace chat template with detected content format: string
[2026-08-27 05:00:32] Auto-detected template features: reasoning_parser=deepseek-r1, tool_call_parser=deepseekv31
[2026-08-27 05:00:42 TP1] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:42 TP1] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:42 TP2] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:42 TP1] Init torch distributed begin.
[2026-08-27 05:00:42 TP2] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:42 TP2] Init torch distributed begin.
[2026-08-27 05:00:42 TP0] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:42 TP0] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:43 TP3] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:43 TP3] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:43 TP0] Init torch distributed begin.
[2026-08-27 05:00:43 TP3] Init torch distributed begin.
[2026-08-27 05:00:43 TP4] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:43 TP4] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:43 TP4] Init torch distributed begin.
[2026-08-27 05:00:44 TP5] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:44 TP5] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:44 TP5] Init torch distributed begin.
[2026-08-27 05:00:44 TP6] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:44 TP6] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:44] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:44] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:44 TP6] Init torch distributed begin.
[2026-08-27 05:00:45 TP7] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:00:45 TP7] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:00:45 TP7] Init torch distributed begin.
[2026-08-27 05:00:45 TP0] sglang is using nccl==2.29.7
[2026-08-27 05:00:48 TP0] All Reduce config: symmetric_memory = 18.01 MB, local_buffer = 8.00 MB, multicast = True
[2026-08-27 05:00:48 TP0] Init torch distributed ends. elapsed=5.38 s, mem usage=1.25 GB
[2026-08-27 05:00:48 TP7] Init torch distributed ends. elapsed=3.12 s, mem usage=1.06 GB
[2026-08-27 05:00:48 TP6] Init torch distributed ends. elapsed=3.63 s, mem usage=1.29 GB
[2026-08-27 05:00:48 TP5] Init torch distributed ends. elapsed=4.16 s, mem usage=1.29 GB
[2026-08-27 05:00:48 TP4] Init torch distributed ends. elapsed=4.61 s, mem usage=1.29 GB
[2026-08-27 05:00:48 TP3] Init torch distributed ends. elapsed=5.10 s, mem usage=1.29 GB
[2026-08-27 05:00:48 TP2] Init torch distributed ends. elapsed=5.74 s, mem usage=1.29 GB
[2026-08-27 05:00:48 TP1] Init torch distributed ends. elapsed=6.10 s, mem usage=1.29 GB
[2026-08-27 05:00:49 TP5] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:49 TP2] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:49 TP4] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:49 TP6] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:49 TP0] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:49 TP1] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:49 TP3] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:49 TP7] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
[2026-08-27 05:00:50 TP0] Load weight begin. avail mem=77.38 GB
[2026-08-27 05:00:50 TP0] Detected fp8 checkpoint.
[2026-08-27 05:00:50 TP4] Load weight begin. avail mem=77.33 GB
[2026-08-27 05:00:50 TP7] Load weight begin. avail mem=77.57 GB
[2026-08-27 05:00:50 TP1] Load weight begin. avail mem=77.33 GB
[2026-08-27 05:00:50 TP5] Load weight begin. avail mem=77.33 GB
[2026-08-27 05:00:50 TP2] Load weight begin. avail mem=77.33 GB
[2026-08-27 05:00:50 TP6] Load weight begin. avail mem=77.33 GB
[2026-08-27 05:00:50 TP3] Load weight begin. avail mem=77.33 GB
[2026-08-27 05:00:50 TP4] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:00:50 TP7] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:00:50 TP1] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:00:50 TP5] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:00:50 TP0] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:00:50 TP2] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:00:50 TP6] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:00:50 TP3] FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=auto, quant_method=KTEPWrapperMethod).
[2026-08-27 05:01:07 TP0] Shared experts fusion optimization enabled.

⏎ Multi-thread loading shards:   0% Completed | 0/163 [00:00<?, ?it/s]
⏎ Multi-thread loading shards:   1% Completed | 1/163 [00:00<01:41,  1.60it/s]
⏎ Multi-thread loading shards:   1% Completed | 2/163 [00:00<00:51,  3.15it/s]
⏎ Multi-thread loading shards:   2% Completed | 4/163 [00:00<00:26,  5.97it/s]
⏎ Multi-thread loading shards:   4% Completed | 6/163 [00:01<00:18,  8.31it/s]
⏎ Multi-thread loading shards:   5% Completed | 8/163 [00:01<00:14, 10.44it/s]
⏎ Multi-thread loading shards:   6% Completed | 10/163 [00:01<00:12, 11.83it/s]
⏎ Multi-thread loading shards:   8% Completed | 13/163 [00:01<00:10, 14.20it/s]
⏎ Multi-thread loading shards:   9% Completed | 15/163 [00:01<00:10, 14.33it/s]
⏎ Multi-thread loading shards:  10% Completed | 17/163 [00:01<00:09, 14.84it/s]
⏎ Multi-thread loading shards:  12% Completed | 19/163 [00:01<00:09, 15.60it/s]
⏎ Multi-thread loading shards:  13% Completed | 21/163 [00:01<00:09, 15.50it/s]
⏎ Multi-thread loading shards:  14% Completed | 23/163 [00:02<00:09, 15.38it/s]
⏎ Multi-thread loading shards:  15% Completed | 25/163 [00:02<00:08, 16.03it/s]
⏎ Multi-thread loading shards:  17% Completed | 27/163 [00:02<00:08, 16.44it/s]
⏎ Multi-thread loading shards:  18% Completed | 29/163 [00:02<00:08, 16.27it/s]
⏎ Multi-thread loading shards:  19% Completed | 31/163 [00:02<00:16,  7.97it/s]
⏎ Multi-thread loading shards:  20% Completed | 33/163 [00:03<00:13,  9.44it/s]
⏎ Multi-thread loading shards:  23% Completed | 37/163 [00:03<00:09, 13.63it/s]
⏎ Multi-thread loading shards:  25% Completed | 40/163 [00:03<00:07, 16.34it/s]
⏎ Multi-thread loading shards:  26% Completed | 43/163 [00:03<00:11, 10.42it/s]
⏎ Multi-thread loading shards:  28% Completed | 45/163 [00:04<00:10, 10.84it/s]
⏎ Multi-thread loading shards:  29% Completed | 47/163 [00:04<00:09, 12.03it/s]
⏎ Multi-thread loading shards:  30% Completed | 49/163 [00:04<00:08, 13.02it/s]
⏎ Multi-thread loading shards:  31% Completed | 51/163 [00:04<00:08, 13.62it/s]
⏎ Multi-thread loading shards:  33% Completed | 53/163 [00:04<00:07, 14.10it/s]
⏎ Multi-thread loading shards:  34% Completed | 55/163 [00:04<00:07, 14.39it/s]
⏎ Multi-thread loading shards:  35% Completed | 57/163 [00:04<00:06, 15.61it/s]
⏎ Multi-thread loading shards:  36% Completed | 59/163 [00:04<00:06, 15.50it/s]
⏎ Multi-thread loading shards:  37% Completed | 61/163 [00:04<00:06, 15.66it/s]
⏎ Multi-thread loading shards:  39% Completed | 63/163 [00:05<00:12,  7.72it/s]
⏎ Multi-thread loading shards:  40% Completed | 65/163 [00:05<00:10,  9.11it/s]
⏎ Multi-thread loading shards:  41% Completed | 67/163 [00:05<00:09, 10.37it/s]
⏎ Multi-thread loading shards:  42% Completed | 69/163 [00:05<00:07, 11.75it/s]
⏎ Multi-thread loading shards:  44% Completed | 71/163 [00:06<00:07, 12.96it/s]
⏎ Multi-thread loading shards:  45% Completed | 73/163 [00:06<00:06, 13.16it/s]
⏎ Multi-thread loading shards:  46% Completed | 75/163 [00:06<00:06, 13.30it/s]
⏎ Multi-thread loading shards:  47% Completed | 77/163 [00:06<00:06, 13.96it/s]
⏎ Multi-thread loading shards:  49% Completed | 80/163 [00:06<00:05, 15.50it/s]
⏎ Multi-thread loading shards:  50% Completed | 82/163 [00:06<00:05, 15.45it/s]
⏎ Multi-thread loading shards:  52% Completed | 84/163 [00:06<00:05, 15.29it/s]
⏎ Multi-thread loading shards:  53% Completed | 86/163 [00:07<00:05, 14.86it/s]
⏎ Multi-thread loading shards:  54% Completed | 88/163 [00:07<00:04, 15.46it/s]
⏎ Multi-thread loading shards:  55% Completed | 90/163 [00:07<00:04, 15.69it/s]
⏎ Multi-thread loading shards:  56% Completed | 92/163 [00:07<00:04, 15.50it/s]
⏎ Multi-thread loading shards:  58% Completed | 94/163 [00:07<00:04, 14.93it/s]
⏎ Multi-thread loading shards:  59% Completed | 96/163 [00:07<00:04, 15.33it/s]
⏎ Multi-thread loading shards:  60% Completed | 98/163 [00:07<00:04, 15.59it/s]
⏎ Multi-thread loading shards:  62% Completed | 101/163 [00:07<00:03, 17.86it/s]
⏎ Multi-thread loading shards:  64% Completed | 104/163 [00:08<00:06,  8.99it/s]
⏎ Multi-thread loading shards:  66% Completed | 107/163 [00:08<00:04, 11.37it/s][2026-08-27 05:01:16 TP4] Load weight end. elapsed=26.47 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.
[2026-08-27 05:01:16 TP5] Load weight end. elapsed=26.47 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.
[2026-08-27 05:01:16 TP6] Load weight end. elapsed=26.51 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.
[2026-08-27 05:01:16 TP3] Load weight end. elapsed=26.54 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.

⏎ Multi-thread loading shards:  67% Completed | 109/163 [00:09<00:06,  8.11it/s]
⏎ Multi-thread loading shards:  68% Completed | 111/163 [00:09<00:05,  8.83it/s][2026-08-27 05:01:17 TP7] Load weight end. elapsed=26.98 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.34 GB, mem usage=3.23 GB.
[2026-08-27 05:01:17 TP2] Load weight end. elapsed=26.99 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.

⏎ Multi-thread loading shards:  69% Completed | 113/163 [00:09<00:04, 10.21it/s]
⏎ Multi-thread loading shards:  71% Completed | 115/163 [00:09<00:04, 11.47it/s]
⏎ Multi-thread loading shards:  72% Completed | 117/163 [00:09<00:03, 12.37it/s][2026-08-27 05:01:17 TP1] Load weight end. elapsed=27.33 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.10 GB, mem usage=3.23 GB.

⏎ Multi-thread loading shards:  73% Completed | 119/163 [00:09<00:03, 13.15it/s]
⏎ Multi-thread loading shards:  74% Completed | 121/163 [00:09<00:03, 13.71it/s]
⏎ Multi-thread loading shards:  76% Completed | 124/163 [00:10<00:02, 15.35it/s]
⏎ Multi-thread loading shards:  77% Completed | 126/163 [00:10<00:02, 15.50it/s]
⏎ Multi-thread loading shards:  79% Completed | 128/163 [00:10<00:02, 15.69it/s]
⏎ Multi-thread loading shards:  80% Completed | 130/163 [00:10<00:02, 15.58it/s]
⏎ Multi-thread loading shards:  81% Completed | 132/163 [00:10<00:01, 16.14it/s]
⏎ Multi-thread loading shards:  82% Completed | 134/163 [00:10<00:01, 16.13it/s]
⏎ Multi-thread loading shards:  83% Completed | 136/163 [00:10<00:01, 15.97it/s]
⏎ Multi-thread loading shards:  85% Completed | 138/163 [00:10<00:01, 15.73it/s]
⏎ Multi-thread loading shards:  86% Completed | 140/163 [00:11<00:01, 16.34it/s]
⏎ Multi-thread loading shards:  87% Completed | 142/163 [00:11<00:01, 16.66it/s]
⏎ Multi-thread loading shards:  88% Completed | 144/163 [00:11<00:01, 16.34it/s]
⏎ Multi-thread loading shards:  90% Completed | 146/163 [00:11<00:01, 16.52it/s]
⏎ Multi-thread loading shards:  91% Completed | 148/163 [00:11<00:00, 16.53it/s]
⏎ Multi-thread loading shards:  92% Completed | 150/163 [00:12<00:01,  7.33it/s]
⏎ Multi-thread loading shards:  93% Completed | 152/163 [00:12<00:01,  8.73it/s]
⏎ Multi-thread loading shards:  94% Completed | 154/163 [00:12<00:00, 10.16it/s]
⏎ Multi-thread loading shards:  96% Completed | 157/163 [00:12<00:00, 12.76it/s]
⏎ Multi-thread loading shards:  98% Completed | 160/163 [00:12<00:00, 15.21it/s]
⏎ Multi-thread loading shards: 100% Completed | 163/163 [00:12<00:00, 12.81it/s]
CPUInfer[0x1b48ced0]: Hello
WorkerPool[0x16bae510] 2 subpools, [numa:threads][0:48] [1:48] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 48 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 48 threads
TP MOE layer 3, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 4, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 5, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 6, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 7, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 8, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 9, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 10, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 11, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 12, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 13, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 14, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 15, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 16, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 17, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 18, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 19, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 20, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 21, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 22, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 23, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 24, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 25, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 26, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 27, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 28, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 29, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 30, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 31, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 32, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 33, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 34, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 35, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 36, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 37, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 38, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 39, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 40, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 41, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 42, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 43, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 44, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 45, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 46, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 1 at numa 1
Creating AMX_MOE_TP 0 at numa 0
TP Load from loader
TP MOE layer 47, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 48, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 49, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 50, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 51, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP MOE layer 52, pool: 0x16bae510, expert num: 257, num_experts_per_tok: 9
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
TP Load from loader
TP [2026-08-27 05:01:46 TP0] Load weight end. elapsed=55.99 s, type=DeepseekV3ForCausalLM, quant=fp8, fmt=e4m3, avail mem=74.15 GB, mem usage=3.23 GB.
[2026-08-27 05:01:48 TP0] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP0] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:48 TP3] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP2] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP7] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP4] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP1] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP5] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP3] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:48 TP6] Fixing v5 tokenizer component mismatch for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52: pre_tokenizer Metaspace -> Sequence, decoder Sequence -> ByteLevel
[2026-08-27 05:01:48 TP2] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:48 TP7] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:48 TP4] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:48 TP1] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:48 TP5] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:48 TP6] Restoring add_bos_token=True for /models/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 (was False after v5 loading)
[2026-08-27 05:01:49 TP7] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP7] Memory pool end. avail mem=11.44 GB
[2026-08-27 05:01:49 TP1] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP1] Memory pool end. avail mem=11.20 GB
[2026-08-27 05:01:49 TP3] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP6] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP3] Memory pool end. avail mem=11.20 GB
[2026-08-27 05:01:49 TP6] Memory pool end. avail mem=11.20 GB
[2026-08-27 05:01:49 TP2] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP2] Memory pool end. avail mem=11.20 GB
[2026-08-27 05:01:49 TP5] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP5] Memory pool end. avail mem=11.20 GB
[2026-08-27 05:01:49 TP4] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 932468, KV size: 61.03 GB
[2026-08-27 05:01:49 TP4] Memory pool end. avail mem=11.20 GB
[2026-08-27 05:01:49 TP0] Memory pool end. avail mem=11.25 GB
[2026-08-27 05:01:49 TP1] [SymmDeviceMemory] Rank: 1, Group size: 8, device_idx: 1, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP0] [SymmDeviceMemory] Rank: 0, Group size: 8, device_idx: 0, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP2] [SymmDeviceMemory] Rank: 2, Group size: 8, device_idx: 2, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP7] [SymmDeviceMemory] Rank: 7, Group size: 8, device_idx: 7, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP6] [SymmDeviceMemory] Rank: 6, Group size: 8, device_idx: 6, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP3] [SymmDeviceMemory] Rank: 3, Group size: 8, device_idx: 3, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP4] [SymmDeviceMemory] Rank: 4, Group size: 8, device_idx: 4, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP5] [SymmDeviceMemory] Rank: 5, Group size: 8, device_idx: 5, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP0] [SymmDeviceMemory] Rank: 0, Group size: 8, device_idx: 0, Signal pad offset: 8192
[2026-08-27 05:01:49 TP5] [SymmDeviceMemory] Rank: 5, Group size: 8, device_idx: 5, Signal pad offset: 8192
[2026-08-27 05:01:49 TP1] [SymmDeviceMemory] Rank: 1, Group size: 8, device_idx: 1, Signal pad offset: 8192
[2026-08-27 05:01:49 TP3] [SymmDeviceMemory] Rank: 3, Group size: 8, device_idx: 3, Signal pad offset: 8192
[2026-08-27 05:01:49 TP2] [SymmDeviceMemory] Rank: 2, Group size: 8, device_idx: 2, Signal pad offset: 8192
[2026-08-27 05:01:49 TP6] [SymmDeviceMemory] Rank: 6, Group size: 8, device_idx: 6, Signal pad offset: 8192
[2026-08-27 05:01:49 TP4] [SymmDeviceMemory] Rank: 4, Group size: 8, device_idx: 4, Signal pad offset: 8192
[2026-08-27 05:01:49 TP7] [SymmDeviceMemory] Rank: 7, Group size: 8, device_idx: 7, Signal pad offset: 8192
[2026-08-27 05:01:49 TP0] [SymmDeviceMemory] Rank: 0, Group size: 8, device_idx: 0, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP6] [SymmDeviceMemory] Rank: 6, Group size: 8, device_idx: 6, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP3] [SymmDeviceMemory] Rank: 3, Group size: 8, device_idx: 3, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP1] [SymmDeviceMemory] Rank: 1, Group size: 8, device_idx: 1, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP4] [SymmDeviceMemory] Rank: 4, Group size: 8, device_idx: 4, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP5] [SymmDeviceMemory] Rank: 5, Group size: 8, device_idx: 5, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP2] [SymmDeviceMemory] Rank: 2, Group size: 8, device_idx: 2, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP7] [SymmDeviceMemory] Rank: 7, Group size: 8, device_idx: 7, Signal pad offset: 704643072
/usr/local/lib/python3.12/dist-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  return func(*args, **kwargs)
[2026-08-27 05:01:49 TP4] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=4, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP5] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=5, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP3] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=3, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP0] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=0, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP1] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=1, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP6] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=6, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP2] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=2, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP7] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=7, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP1] [SymmDeviceMemory] Rank: 1, Group size: 8, device_idx: 1, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP0] [SymmDeviceMemory] Rank: 0, Group size: 8, device_idx: 0, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP7] [SymmDeviceMemory] Rank: 7, Group size: 8, device_idx: 7, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP2] [SymmDeviceMemory] Rank: 2, Group size: 8, device_idx: 2, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP6] [SymmDeviceMemory] Rank: 6, Group size: 8, device_idx: 6, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP3] [SymmDeviceMemory] Rank: 3, Group size: 8, device_idx: 3, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP5] [SymmDeviceMemory] Rank: 5, Group size: 8, device_idx: 5, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP4] [SymmDeviceMemory] Rank: 4, Group size: 8, device_idx: 4, Signal pad offset: 234881024
[2026-08-27 05:01:49 TP3] [SymmDeviceMemory] Rank: 3, Group size: 8, device_idx: 3, Signal pad offset: 8192
[2026-08-27 05:01:49 TP0] [SymmDeviceMemory] Rank: 0, Group size: 8, device_idx: 0, Signal pad offset: 8192
[2026-08-27 05:01:49 TP7] [SymmDeviceMemory] Rank: 7, Group size: 8, device_idx: 7, Signal pad offset: 8192
[2026-08-27 05:01:49 TP4] [SymmDeviceMemory] Rank: 4, Group size: 8, device_idx: 4, Signal pad offset: 8192
[2026-08-27 05:01:49 TP5] [SymmDeviceMemory] Rank: 5, Group size: 8, device_idx: 5, Signal pad offset: 8192
[2026-08-27 05:01:49 TP1] [SymmDeviceMemory] Rank: 1, Group size: 8, device_idx: 1, Signal pad offset: 8192
[2026-08-27 05:01:49 TP2] [SymmDeviceMemory] Rank: 2, Group size: 8, device_idx: 2, Signal pad offset: 8192
[2026-08-27 05:01:49 TP6] [SymmDeviceMemory] Rank: 6, Group size: 8, device_idx: 6, Signal pad offset: 8192
[2026-08-27 05:01:49 TP0] [SymmDeviceMemory] Rank: 0, Group size: 8, device_idx: 0, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP1] [SymmDeviceMemory] Rank: 1, Group size: 8, device_idx: 1, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP3] [SymmDeviceMemory] Rank: 3, Group size: 8, device_idx: 3, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP2] [SymmDeviceMemory] Rank: 2, Group size: 8, device_idx: 2, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP6] [SymmDeviceMemory] Rank: 6, Group size: 8, device_idx: 6, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP5] [SymmDeviceMemory] Rank: 5, Group size: 8, device_idx: 5, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP7] [SymmDeviceMemory] Rank: 7, Group size: 8, device_idx: 7, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP4] [SymmDeviceMemory] Rank: 4, Group size: 8, device_idx: 4, Signal pad offset: 704643072
[2026-08-27 05:01:49 TP1] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=1, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP0] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=0, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP3] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=3, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP2] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=2, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP5] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=5, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP6] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=6, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP4] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=4, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP7] FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, rank=7, world_size=8, max_token_num=2048, hidden_dim=7168
[2026-08-27 05:01:49 TP0] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP7] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP6] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP5] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP4] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP3] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP2] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP1] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
[2026-08-27 05:01:49 TP0] max_total_num_tokens=932468, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=2913, context_len=163840, available_gpu_mem=8.57 GB
[2026-08-27 05:01:49 TP0] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:49 TP7] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:49 TP6] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:49 TP5] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:49 TP4] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:49 TP3] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:49 TP1] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:49 TP2] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False streaming_wrapped=False
[2026-08-27 05:01:50] Engine startup timings (s): load_weight=55.99, kv_cache_allocation=0.15, scheduler_e2e=70.81, cuda_graph={prefill=0.00, decode=0.00, target_verify=0.00, draft_prefill=0.00, draft_decode=0.00, draft_extend=0.00}, tokenizer_e2e=84.72
[2026-08-27 05:01:50] INFO:     Started server process [12724]
[2026-08-27 05:01:50] INFO:     Waiting for application startup.
[2026-08-27 05:01:50] Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_p': 0.95}
[2026-08-27 05:01:50] INFO:     Application startup complete.
[2026-08-27 05:01:50] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2026-08-27 05:01:51] INFO:     127.0.0.1:41166 - "GET /model_info HTTP/1.1" 200 OK
[2026-08-27 05:01:52 TP5] Disable CP decode attention TP
[2026-08-27 05:01:52 TP7] Disable CP decode attention TP
[2026-08-27 05:01:52 TP6] Disable CP decode attention TP
[2026-08-27 05:01:52 TP4] Disable CP decode attention TP
[2026-08-27 05:01:52 TP2] Disable CP decode attention TP
[2026-08-27 05:01:52 TP3] Disable CP decode attention TP
[2026-08-27 05:01:52 TP1] Disable CP decode attention TP
[2026-08-27 05:01:52 TP0] Disable CP decode attention TP
[2026-08-27 05:01:52 TP4] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP7] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP6] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP2] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP0] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP5] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP3] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP1] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP5] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP6] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP4] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP0] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP1] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP2] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP3] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:52 TP7] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json
[2026-08-27 05:01:54] INFO:     127.0.0.1:44208 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 05:02:02 TP0] Using configuration from /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=4096,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json for W8A8 Block FP8 kernel.
[2026-08-27 05:02:03 TP5] Triton kernel '_fwd_kernel' took 1.00 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:03 TP6] Triton kernel '_fwd_kernel' took 1.00 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:03 TP0] Triton kernel '_fwd_kernel' took 1.01 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:03 TP0] Using configuration from /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=7168,K=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json for W8A8 Block FP8 kernel.
[2026-08-27 05:02:03 TP4] Triton kernel '_fwd_kernel' took 1.02 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:03 TP1] Triton kernel '_fwd_kernel' took 1.01 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:03 TP7] Triton kernel '_fwd_kernel' took 1.00 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:03 TP0] Using configuration from /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=4608,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json for W8A8 Block FP8 kernel.
[2026-08-27 05:02:04 TP0] Using configuration from /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs/N=7168,K=2304,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json for W8A8 Block FP8 kernel.
[2026-08-27 05:02:04] INFO:     127.0.0.1:41992 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 05:02:14] INFO:     127.0.0.1:45788 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 05:02:24] INFO:     127.0.0.1:38930 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 05:02:34] INFO:     127.0.0.1:56728 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 05:02:37 TP0] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:37 TP0] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP1] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP1] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP2] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP2] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP3] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP3] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP7] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP7] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP6] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP6] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP5] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP5] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP4] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:38 TP4] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=0,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
[2026-08-27 05:02:44 TP0] Prefill batch, #new-seq: 1, #new-token: 7, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.13
[2026-08-27 05:02:44] INFO:     127.0.0.1:36664 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-08-27 05:02:46] INFO:     127.0.0.1:41182 - "POST /generate HTTP/1.1" 200 OK
[2026-08-27 05:02:46 TP7] Freezing GC in Scheduler process. gen0: 50->0, gen1: 6->0, gen2: 1102452->0
[2026-08-27 05:02:46 TP2] Freezing GC in Scheduler process. gen0: 47->0, gen1: 6->0, gen2: 1103144->0
[2026-08-27 05:02:46 TP4] Freezing GC in Scheduler process. gen0: 39->0, gen1: 29->0, gen2: 1101453->0
[2026-08-27 05:02:46 TP6] Freezing GC in Scheduler process. gen0: 39->0, gen1: 29->0, gen2: 1102834->0
[2026-08-27 05:02:46 TP1] Freezing GC in Scheduler process. gen0: 39->0, gen1: 27->0, gen2: 1101537->0
[2026-08-27 05:02:46 TP5] Freezing GC in Scheduler process. gen0: 39->0, gen1: 29->0, gen2: 1102667->0
[2026-08-27 05:02:46 TP0] Freezing GC in Scheduler process. gen0: 39->0, gen1: 27->0, gen2: 1099236->0
[2026-08-27 05:02:46 TP3] Freezing GC in Scheduler process. gen0: 39->0, gen1: 27->0, gen2: 1103149->0
[2026-08-27 05:02:46] Freezing GC in Detokenizer Manager process. gen0: 46->0, gen1: 5->0, gen2: 851169->0
[2026-08-27 05:02:46] Freezing GC in Tokenizer Manager process. gen0: 320->0, gen1: 1231->0, gen2: 916065->0
[2026-08-27 05:02:46] INFO:     127.0.0.1:36666 - "POST /freeze_gc HTTP/1.1" 200 OK
[2026-08-27 05:02:46] The server is fired up and ready to roll!
[2026-08-27 05:02:56 TP0] Prefill batch, #new-seq: 1, #new-token: 1, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.08
[2026-08-27 05:02:57] INFO:     127.0.0.1:55668 - "GET /health HTTP/1.1" 200 OK
[2026-08-27 05:02:58 TP0] Triton kernel '_fwd_kernel' took 1.73 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:58 TP6] Triton kernel '_fwd_kernel' took 1.74 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:58 TP4] Triton kernel '_fwd_kernel' took 1.74 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:58 TP7] Triton kernel '_fwd_kernel' took 1.75 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:58 TP5] Triton kernel '_fwd_kernel' took 1.75 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:58 TP1] Triton kernel '_fwd_kernel' took 1.76 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:59 TP3] Triton kernel '_fwd_kernel' took 1.87 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:02:59 TP2] Triton kernel '_fwd_kernel' took 1.88 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:01 TP0] Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 3, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.62
[2026-08-27 05:03:10 TP0] Decode batch, #running-req: 1, #token: 0, token usage: 0.00, cuda graph: False, gen throughput (token/s): 0.50, #queue-req: 0
[2026-08-27 05:03:10] INFO:     127.0.0.1:55676 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:25] INFO:     127.0.0.1:35846 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 05:03:25] INFO:     127.0.0.1:35846 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 05:03:31 TP0] Triton kernel '_fwd_kernel' took 1.73 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:31 TP6] Triton kernel '_fwd_kernel' took 1.74 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:31 TP2] Triton kernel '_fwd_kernel' took 1.74 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:31 TP7] Triton kernel '_fwd_kernel' took 1.74 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:31 TP3] Triton kernel '_fwd_kernel' took 1.75 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:31 TP1] Triton kernel '_fwd_kernel' took 1.75 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:31 TP4] Triton kernel '_fwd_kernel' took 1.76 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:31 TP5] Triton kernel '_fwd_kernel' took 1.76 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.
[2026-08-27 05:03:43 TP0] Prefill batch, #new-seq: 1, #new-token: 1022, #cached-token: 1, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 24.13
[2026-08-27 05:03:43] INFO:     127.0.0.1:35846 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:43 TP0] Prefill batch, #new-seq: 7, #new-token: 7108, #cached-token: 7, token usage: 0.01, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 16161.82
[2026-08-27 05:03:43] INFO:     127.0.0.1:35862 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:43] INFO:     127.0.0.1:35868 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:43] INFO:     127.0.0.1:35884 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:43] INFO:     127.0.0.1:35892 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:43] INFO:     127.0.0.1:35902 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:43] INFO:     127.0.0.1:35906 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:43] INFO:     127.0.0.1:35908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:03:55 TP0] Decode batch, #running-req: 8, #token: 7071, token usage: 0.01, cuda graph: False, gen throughput (token/s): 6.86, #queue-req: 0
[2026-08-27 05:04:07 TP0] Decode batch, #running-req: 8, #token: 7391, token usage: 0.01, cuda graph: False, gen throughput (token/s): 26.50, #queue-req: 0
[2026-08-27 05:04:20 TP0] Decode batch, #running-req: 8, #token: 7711, token usage: 0.01, cuda graph: False, gen throughput (token/s): 26.02, #queue-req: 0
[2026-08-27 05:04:32 TP0] Prefill batch, #new-seq: 4, #new-token: 3288, #cached-token: 792, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 68.18
[2026-08-27 05:04:32] INFO:     127.0.0.1:35884 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:32] INFO:     127.0.0.1:35892 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:32] INFO:     127.0.0.1:35908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:32] INFO:     127.0.0.1:35862 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:34 TP0] Prefill batch, #new-seq: 4, #new-token: 3256, #cached-token: 792, token usage: 0.01, #running-req: 4, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 1483.07
[2026-08-27 05:04:34] INFO:     127.0.0.1:35902 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:34] INFO:     127.0.0.1:35906 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:34] INFO:     127.0.0.1:35868 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:34] INFO:     127.0.0.1:35846 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:04:43 TP0] Decode batch, #running-req: 8, #token: 6997, token usage: 0.01, cuda graph: False, gen throughput (token/s): 13.54, #queue-req: 0
[2026-08-27 05:04:55 TP0] Decode batch, #running-req: 8, #token: 7317, token usage: 0.01, cuda graph: False, gen throughput (token/s): 26.38, #queue-req: 0
[2026-08-27 05:05:08 TP0] Decode batch, #running-req: 8, #token: 7637, token usage: 0.01, cuda graph: False, gen throughput (token/s): 26.39, #queue-req: 0
[2026-08-27 05:05:22 TP0] Prefill batch, #new-seq: 6, #new-token: 4886, #cached-token: 1189, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 101.17
[2026-08-27 05:05:22] INFO:     127.0.0.1:35884 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:22] INFO:     127.0.0.1:35892 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:22] INFO:     127.0.0.1:35908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:22] INFO:     127.0.0.1:35862 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:22] INFO:     127.0.0.1:35902 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:22] INFO:     127.0.0.1:35906 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:24 TP0] Prefill batch, #new-seq: 2, #new-token: 1610, #cached-token: 396, token usage: 0.01, #running-req: 6, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 1147.57
[2026-08-27 05:05:24] INFO:     127.0.0.1:35868 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:24] INFO:     127.0.0.1:35846 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:05:31 TP0] Decode batch, #running-req: 8, #token: 6887, token usage: 0.01, cuda graph: False, gen throughput (token/s): 13.88, #queue-req: 0
[2026-08-27 05:05:43 TP0] Decode batch, #running-req: 8, #token: 7207, token usage: 0.01, cuda graph: False, gen throughput (token/s): 25.14, #queue-req: 0
[2026-08-27 05:05:56 TP0] Decode batch, #running-req: 8, #token: 7527, token usage: 0.01, cuda graph: False, gen throughput (token/s): 25.28, #queue-req: 0
[2026-08-27 05:06:11 TP0] Prefill batch, #new-seq: 1, #new-token: 820, #cached-token: 199, token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 17.18
[2026-08-27 05:06:11] INFO:     127.0.0.1:35884 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:15 TP0] Prefill batch, #new-seq: 7, #new-token: 5756, #cached-token: 1388, token usage: 0.01, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 1635.20
[2026-08-27 05:06:15] INFO:     127.0.0.1:35892 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:15] INFO:     127.0.0.1:35908 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:15] INFO:     127.0.0.1:35862 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:15] INFO:     127.0.0.1:35902 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:15] INFO:     127.0.0.1:35906 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:15] INFO:     127.0.0.1:35868 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:15] INFO:     127.0.0.1:35846 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-27 05:06:20 TP0] Decode batch, #running-req: 8, #token: 6904, token usage: 0.01, cuda graph: False, gen throughput (token/s): 13.58, #queue-req: 0
[2026-08-27 05:06:32 TP0] Decode batch, #running-req: 8, #token: 7224, token usage: 0.01, cuda graph: False, gen throughput (token/s): 26.16, #queue-req: 0
[2026-08-27 05:06:44 TP0] Decode batch, #running-req: 8, #token: 7544, token usage: 0.01, cuda graph: False, gen throughput (token/s): 26.40, #queue-req: 0
[2026-08-27 05:06:54] INFO:     127.0.0.1:35884 - "GET /metrics HTTP/1.1" 404 Not Found
[2026-08-27 05:06:54] INFO:     127.0.0.1:35892 - "GET /metrics HTTP/1.1" 404 Not Found
```

## 32. `eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/smoke.json`

- 바이트 **439** · 줄 **1** · SHA256 `0e80ea9a2fdaea731459f7cd3f95f81d94e1767d41ac26f9e7e08a343232ba36`
- 인코딩 utf-8

정리한 형태:

```json
{
  "id": "39daf2f457c0421f8128a610311a4847",
  "object": "text_completion",
  "created": 1787806990,
  "model": "r1",
  "choices": [
    {
      "index": 0,
      "text": " _____________?\nA..' Paris''\nB..'':opposite:''''- Orleans''''uzz'''')''''lius''''",
      "logprobs": null,
      "finish_reason": "length",
      "matched_stop": null
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 38,
    "completion_tokens": 32,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

원문:

```text
{"id":"39daf2f457c0421f8128a610311a4847","object":"text_completion","created":1787806990,"model":"r1","choices":[{"index":0,"text":" _____________?\nA..' Paris''\nB..'':opposite:''''- Orleans''''uzz'''')''''lius''''","logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":6,"total_tokens":38,"completion_tokens":32,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
```

