# EPOCH 검증 캠페인 (2026-09-10 ~ 09-11) — 자료 색인

사용자가 반입한 실행계획서 `plan/epoch_experiment_plan_20260909.md` 의 EXP-E00~E12 를 수행하고,
그 과정에서 파생된 진단을 함께 기록한 캠페인의 전 자료다.

- **결과 요약과 다른 세션으로의 인계**: [`HANDOFF.md`](HANDOFF.md) ← 먼저 읽을 것
- 실행계획서·사양서: `plan/`
- 측정 원본 (RUN.log, 벤치 로그, 서버 로그, DRAM 추적): `results/`
- 실행 스크립트 (하네스) 와 소스 패치: `harness/`

## 대상 시스템

| 항목 | 값 |
|---|---|
| 모델 | Qwen3-Coder-480B-A35B-Instruct FP8 (GPU) + INT4 expert (CPU, AMXINT4) |
| 서빙 | SGLang + KTransformers kt-kernel 0.7.0.post1 (컨테이너 `sgl-kt`) |
| GPU | NVIDIA H100 80GB × 4 (TP=4) |
| CPU | Intel Xeon 8480+ × 2 (96 워커, 2 풀, turbo off 2.0 GHz) |
| 메모리 | DDR5-4400 8채널 × 2DPC / 소켓 (이론 282 GB/s/소켓) |
| 벤치 클라이언트 | 컨테이너 `vllm-h100`, `vllm bench serve` |

## 측정 규율

- 모든 벤치 직전 `POST /flush_cache` + 3초 대기 (벤치 **간** 접두 캐시 오염 제거)
- 고정 작업량 비교는 요청 수를 고정 (512 또는 2,048)
- open-loop 은 `--max-concurrency` 미사용
- 물리 DRAM 은 호스트 `perf` uncore_imc (소켓당 8채널, `cas_count_read/write` × 64 B),
  정상상태 안쪽 창만 사용

## 결과 디렉토리 (시간순)

| 디렉토리 | 내용 | 문서 |
|---|---|---|
| `20260910_002116_expE00_E01_E02` | E00~E02 1차 (flush 도입 전, 오염분) | RUN.log |
| `20260910_002544_expE00_E01_E02` | E00 구성고정 / E01 재현 / E02 KV×graph 2×2 + capacity cap | RESULTS.md |
| `20260910_055709_expE06_E07` | E06 최대 처리량·지연 경계 / E07 동일 SLO 용량 | RESULTS.md |
| `20260910_070926_expE03_E07ops_E09` | E07-ops / E03 논리 대리지표 / E09 품질 | RESULTS.md |
| `20260910_121248_expE04_E05_E08_E10` | E04 CPU 워커 / E05 E×R 2×2 / E08 mixed / E10 워크로드 | RESULTS.md |
| `20260910_155436_expE12_soak_reps` | **E12** 반복 신뢰구간 + 60분 soak 3구성 | RESULTS.md, ANALYSIS.md, DRAM.md |
| `20260910_191714_e03_physical_dram` | **E03 §9.4** 물리 DRAM (동일 도착열 3구성) | RESULTS.md |
| `20260910_194924_e11_stage1_chunkcal` | E11 1단계 chunk·eager 벌점 보정 | RUN.log |
| `20260910_201658_ide063_nan_isolation` | NaN 원인 분리 4셀 | RESULTS.md |
| `20260910_210444_e03fix_dram_fixedC` | E03 보정 (계측 창 결함 수정) | RESULTS.md |
| `20260910_213017_ide063b_nan_history` | NaN 누적 이력 재현 | RUN.log |
| `20260910_220047_ide061_hsweep` | hot expert 수 H 스윕 1단계 | RESULTS.md |
| `20260910_223414_ide062_lowload_tpot` | 저부하 지연 원인 판별 4셀 | RESULTS.md |
| `20260910_230420_ide064_nan_mitigation` | NaN 완화 수단 검증 | RESULTS.md |
| `20260910_235804_ide061b_hsweep_uncapped` | H 스윕 2단계 (KV 상한 해제) | RESULTS.md |
| `20260911_003216_e11_stage3_oracle` | **E11 3단계** held-out 전수측정 24셀 | RESULTS.md, JUDGMENT.md |
| `20260911_003216_ide061c_h112` | H=112 재시도 (메모리 불가) | RUN.log |
| `20260911_042249_e11_wh3_graphaxis` | E11 보강 WH3 (graph 축 재판별) | RESULTS.md |
| `20260911_064448_e12redo_epops_soak` | **E12 재측정** (완화 적용, 미달 게이트 해소) | RESULTS.md |
| `20260911_075259_ide065_bucket_spacing` | 버킷 간격 1차 | RESULTS.md |
| `20260911_082058_ide065b_spacing_reps` | 버킷 간격 2차 (반복 확정) | RESULTS.md |
| `20260911_085338_ide066_nan_probe` | NaN 발생 지점 계측 | RESULTS.md |

## 하네스 (`harness/`)

| 파일 | 용도 |
|---|---|
| `run_expE12.sh` | E12 반복 + 60분 soak |
| `run_e03phys.sh`, `run_e03fix.sh` | 물리 DRAM 계측 (2판은 계측 창 수정) |
| `run_e11_stage1.sh`, `run_e11_stage3.sh` | E11 보정 / held-out 전수측정 |
| `run_hsweep.sh` | hot expert 수 스윕 |
| `run_ide062.sh`, `run_ide063.sh` | 저부하 지연 / NaN 원인 분리 |
| `dram_sample.sh` | perf uncore_imc 소켓별 DRAM 샘플러 |
| `align_dram.py` | DRAM 추적을 벤치 구간에 정렬해 토큰당 바이트 산출 |
| `analyze_e12.py` | 반복 통계·soak 열화·이상로그 집계 |
| `collect_cells.py` | 전 측정 셀을 CSV 로 수집 |
| `kt_nan_probe_patch.py` | KTransformers MoE 합산 지점 NaN 계측 (1판, 적용됨) |
| `kt_nan_probe_patch2.py` | 같은 계측 2판 (전 랭크 + 심장박동). **작성만 했고 실행하지 않음** |

## 비용모델 코드

`shadow_assists/features/IDE_060/model/` 에 있다 (`selector.py`, `make_predictions.py`,
`analyze_e11.py`). 사양서와 학습·평가 분리, 사전등록 예측은 `plan/` 에 사본을 두었다.
