# CPU MoE 오프로딩 실험 결과 요약 — IDE_070 → IDE_071 → IDE_072 (2026-09-15~16)

이 문서는 세 캠페인의 측정값을 한 곳에 모은 결과 요약이다. 캠페인별 원문서(FULL_REPORT/RESULT)는 지시서 규칙에 따라 평가 문장이 없고, 이 문서는 사용자 요청으로 작성한 별도 정리본이다. 모든 수치는 각 캠페인 원자료(`eval/results/IDE_071_20260916/`, `eval/results/IDE_072_20260916/`, `eval/results/*_ide070_*`)에서 가져왔고 파일 경로를 병기한다.

## 0. 공통 조건
- 노드 violet-h100-016: H100 80GB ×4 (TP4, GPU 0–3), Xeon 8480+ ×2 (CPU expert 워커 96 스레드: 물리 코어 0–47, 56–103; turbo OFF 2.0 GHz, BIOS 잠금), DDR5-4400 2 TB.
- 모델 Qwen3-Coder-480B-A35B-Instruct-FP8 (GPU: attention·dense·hot expert FP8), CPU expert = kt INT4 (AMXINT4, decode 핫루프는 AVX-512 VNNI), SGLang 0.5.18 @71de97b + 로컬 수정, kt-kernel 0.7.0.post1 소스 빌드 (IDE_033 callback-free 핸드오프 포함).
- 주 워크로드 **SHORT_COLD**: sonnet 512 입력/128 출력, 동시성 64, 256 요청, ignore_eos, 매 반복 전 KV/prefix 캐시 flush, seed 20260916. (IDE_070 의 642 tok/s 는 flush 없는 LEGACY 프로토콜 값이라 직접 비교 대상이 아니다. 같은 구성의 SHORT_COLD 값은 612.)
- 표기: 평균 ± 표본표준편차 (반복 수). 단위 tok/s = 출력 토큰 처리량.

## 1. 처리량 변화 (SHORT_COLD C64)

| 단계 | 구성 | tok/s | 원자료 |
|---|---|---|---|
| 기준선 R0 | hot96 (hotmap v1), deferred 4, 기본 핸드오프, KV 40,960 bf16, graph ≤64 | **612.1 ± 6.8 (3)** | IDE_071 R00_r0_def4 |
| deferred 0 | R0 에서 deferral 해제 | 578.1 ± 1.3 (3) | IDE_071 R01 |
| deferred 8 | R0 에서 deferral 8 | 643.2 ± 4.5 (3) | IDE_071 P2_cf0_skip0_pin0_def8 |
| callback-free (CF) | + KT_CALLBACK_FREE=1, deferred 8 | 655.0 ± 2.1 (3) | P2_cf1_skip0_pin0_def8 |
| CF + 빈 immediate 생략 | + KT_CF_SKIP_EMPTY_IMM=1 | 736.4 ± 9.5 (3) | P2_cf1_skip1_pin0_def8 |
| + 비-kt 스레드 재배치 (= B3) | 위 + 비-kt 1,197 스레드를 코어 48–55,104–111 과 HT 형제로 | **740.3 ± 10.7 (3)** | P2_cf1_skip1_pin1_def8 |
| hotmap v2 + 층별 비균일 배치 (= S2) | B3 + hotmap_v2 + 층별 GPU expert 수 75~142 (총 5,952 슬롯 = hot96 과 동일 HBM) | **805.8 ± 5.3 (3)** | IDE_071 compact S2_b3_v2_nu5952__confirm |
| S2 재현 (V0) | 새 프로세스 3회 | 799.8 ± 4.1 (3) | IDE_072 V0_confirm |
| + AVX 레지스터 블로킹 커널 (V1) | V0 + KT_AVX_RB=0 (소스상 변수 존재 = ON) | **844.0 ± 14.9 (3)** | IDE_072 V1_confirm |
| V1 별도 입력 | prefix100 seed 20261002 | 887.1 (1) | IDE_072 V1_confirm COMPACT_ALT |
| V1 연속 부하 | 1,024 요청 | 897.8 (1) | IDE_072 V1_load |
| 참조: GPU-only TP8+EP8 (GPU 8장, CPU expert 없음) | KV 131,072 | 1,703.5 ± 4.8 (3) (C32: 1,113.2) | IDE_071 R07 |

기준선 612 → V1 844 = +37.9 % (SHORT_COLD, 같은 GPU 4장). 이 값은 GPU 8장 GPU-only 1,704 의 49.5 %.

## 2. 요인별 결과 (앵커 = B3 740.3 또는 B4, SHORT_COLD C64, IDE_071)

### 2.1 효과가 측정된 요인
| 요인 | 비교 | 결과 |
|---|---|---|
| deferral 0 → 4 → 8 (기본 핸드오프) | 580.3 → 612.4 → 643.2 | 단조 증가 |
| callback-free ON (deferred 8) | 643.2 → 655.0 | +1.8 % |
| 빈 immediate 생략 (CF·deferred 8 에서만 동작) | 655.0 → 736.4 | +12.4 % |
| 비-kt 스레드 재배치 | 736.4 → 740.3 | 오차 범위 |
| hotmap v2 (uniform 96) | R0 612.1 → 626.1 | +2.3 % |
| 층별 비균일 배치 (v2, 5,952 슬롯) | 626.1 → 655.8 (R0 기준) / 740.3 → 805.8 (B3 기준) | +4.7 % / +8.8 % |
| AVX 레지스터 블로킹 (KT_AVX_RB) | B3 740.3 → 787.1 / S2 799.8 → 844.0 | +6.3 % / +5.5 % |
| overlap scheduler OFF | 612.4 → 590.7 (CF OFF), 619.1 → 593.8 (CF ON) | −3~4 % |
| skip-empty 해제 (V2) | V0 795.4 → 702.0 (1회) | −11.7 % |

### 2.2 효과가 없거나 악화된 요인 (B3 앵커 740.3 대비, 3회 평균)
| 요인 | 값 | 결과 |
|---|---|---|
| CPU 워커 수 80 / 88 / 104 / 108 | 698.2 / 727.5 / 735.0 / 742.0 | 96 이하는 감소, 초과는 오차 범위 (112 는 부팅 실패) |
| AVX prefetch 0/1/2/4, KT_FUSE_QIN, AMX_MIN_ROWS 1~16 | 732~760 | 오차 범위 |
| AMX_MIN_QLEN 32 / 1,000,000 | 676.5 / 677.2 | −8.5 % (기본 80 대비) |
| OMP/MKL/OPENBLAS 스레드 1·4, 토크나이저 워커 1·2·4 | 724~757 | 오차 범위 |
| NUMA interleave / membind0 (R0 기준 612) | 656.4 / 649.7 (LEGACY 프로토콜, IDE_070 B) | LEGACY 값이라 별도 표기 |
| threadpool 1 / socket0 전용 (IDE_070) | 450 / 430 | 크게 악화 |
| B4 스택 변형 (S1 pin, S3 graph64, S4 KV40960, S5 chunk2048) | 716.9 / 707.6 / 703.8 / 620.8 (B4 705.1) | pin +1.7 %, 나머지 0 또는 악화 |
| 9-10 캠페인 스택 B4 (fp8 KV, KV 143k, mixed hotmap 등) | 705.1 (SHORT_COLD) / 816.9 (LEGACY) | SHORT_COLD 에서는 B3 계열보다 낮음 |
| AWQ 4-bit GPU expert (hot 96/128/144, graph on/off) | 부팅 실패 5/5 (illegal memory access, moe_sum_reduce 실패) | 측정값 없음 |
| turbo ON | BIOS 잠금 (root 쓰기 거부) | 실행 불가 |

## 3. 정확성 (GSM8K 앞 20문항, greedy, max_tokens 1024, IDE_072)
| 구성 | 정답 | 절단·오류 |
|---|---|---|
| V0 (S2) | 20/20 | 0 |
| V1 (+RB) | 19/20 | 0 |
| V2 (skip-empty OFF) | 20/20 | 0 |
| Q0 (V2 + deferred 0) | 19/20 | 0 |

짝비교: V0–V1 출력 전문 일치 11/20, 추출답 일치 19/20; V0–V2 전문 20/20; V2–Q0 전문 1/20, 추출답 19/20. greedy 4문항 smoke 는 모든 부팅에서 정상. IDE_071 S2 GSM40 (768 tokens) 38/40.

## 4. 안정성·오류
- NaN 크래시 (`probability tensor contains inf/nan` → CUBLAS 실패): IDE_071 에서 2회 (R03 = B3 재현 셀 rep1, S2 연속 부하 a1 워밍업), 모두 콜드 스타트 직후. IDE_072 에서 V0·V1·V2 첫 부팅·재전송(32 요청)·확인 부팅·1,024 요청 연속 부하까지 0회 → `NOT_REPRODUCED_WITHIN_BUDGET`. 원 실패 프롬프트 원문은 미보존.
- 층별 배치 패치는 서버 로그의 62층 값 합계 5,952 로 매번 확인; per-layer 패치 미적용 실행(IDE_070 1차 nu5952)은 무효 처리했다.

## 5. CPU 측 측정 (IDE_070, 기준선 R0)
- DRAM 대역폭 소켓당 ~107 GB/s (실측 천장 ~215/소켓의 50 %), 원격 NUMA 접근 2 %, 코어 IPC 1.25, Backend bound 67 %.
- kt 워커 perf 샘플: AVX-512 VNNI 커널 36 %, 시계 폴링·스핀·동기화 37 % → 워커 시간의 약 40 % 가 대기.
- cold expert 선택 7.30/토큰 (uniform 96, 1.47 %); 층별 비균일 예산으로 계산상 5.96/토큰.
- 이 측정은 이번 개선(배치·핸드오프·커널) 이후 다시 하지 않았다.

## 6. 남은 것
1. V1(RB 커널) 구성의 정확성 확인은 20문항뿐이다.
2. NaN 크래시는 원인·재현 입력이 없다.
3. CPU 워커 대기 40 %·DRAM 50 % 는 층 단위 동기 호출 구조에서 오는 것으로, 설정 탐색 범위 밖(코드 공사)이다.

## 7. 원자료·문서
- IDE_070: `shadow_assists/features/IDE_070/RESULT.md` (커밋 448aad450)
- IDE_071: `shadow_assists/features/IDE_071/{FULL_REPORT.md, RESULT.md, COMPLETION_STATUS.md}` (data commit 5227c164)
- IDE_072: `shadow_assists/features/IDE_072/{FULL_REPORT.md, RESULT.md, quality/paired_question_results.md, failures/README.md}` (data commit 1579f567)
- 하네스: `eval/ide070/`, `eval/ide071/`
