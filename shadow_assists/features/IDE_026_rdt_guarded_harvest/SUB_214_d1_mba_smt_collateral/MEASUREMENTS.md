# SUB_214 [D1] — MEASUREMENTS (확정판, 2026-06-12)

> **판정 요약**: ① `thread_throttle_mode=max` **연좌 throttle 은 vLLM 실부하에서 미발생**
> (C3 ≈ C0, C3/C2 = +18%) — D1 게이트 (연좌 ≥+20% → 코어-배타 강제) **불충족** =
> 코어-배타 강제 불필요. ② 대신 **"sibling harvest 는 반드시 MBA 가드"** 규칙 채택
> (무가드 −16% → MBA20 가드 −1%). ③ 성능 결론: **MBA 20% 가드 하에 sibling HT
> 16스레드 harvest 슬롯이 serving 손실 ~1% 로 확보** — D2 (SUB_215, HT 빌리기) GO.

## 1. 측정 환경 (SUB_212 canonical 준수 + 통제 변경)

| 항목 | 값 |
|---|---|
| 머신 | dgx-b200 호스트 직접 (2× Xeon 8570 EMR + 8× B200) |
| 모델/설정 | Llama-3.1-70B-Instruct, TP=8, **suffix K=32**, FaP, gmu 0.85, MML 16384 |
| 부하 | conc=32, max_tokens 8192, streaming, 실 trace (`sampled_prompts.parquet`) |
| corpus | sharegpt/swebench/humaneval/mbpp/wildchat/lmsys + mix(500, shuffle seed0) |
| 통제 변경 | vLLM 을 primary HT `0-47,56-103` taskset 고정 (sibling·free 코어 확보) |
| venv/env | 호스트 `vllm_dev_prj`, CUDA_HOME=cuda-13.0, canonical env (TSK_042) |
| **부팅 정책** | **셀마다 fresh boot** (v2) — §4 방법론 발견 때문 |

셀 정의 (aggressor = `victim_aggressor` AVX-512 triad 16스레드, array 32MB/스레드):

| 셀 | aggressor 위치 | harvest CLOS MBA |
|---|---|---|
| C0_base | 없음 | — |
| C1_sep20 | free 물리코어 `48-55,104-111` | 20% |
| C2_sib100 | **vLLM sibling** `112-119,168-175` | 100% (무제한) |
| C3_sib20 | vLLM sibling (동일) | **20%** |

## 2. 본판정 데이터 (v2, 셀별 fresh boot — accept 4셀 일치 0.721~0.727)

| corpus | C0_base | C1_sep20 | C2_sib100 | C3_sib20 | C1/C0 | C2/C0 | C3/C0 | **C3/C2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sharegpt | 4,532 | 4,472 | 3,988 | 4,554 | 0.987 | 0.880 | 1.005 | 1.142 |
| swebench | 4,997 | 4,845 | 4,393 | 5,016 | 0.970 | 0.879 | 1.004 | 1.142 |
| humaneval | 4,945 | 4,631 | 3,709 | 4,483 | 0.937 | 0.750 | 0.907 | 1.209 |
| mbpp | 2,475 | 2,527 | 2,270 | 2,418 | 1.021 | 0.917 | 0.977 | 1.065 |
| wildchat | 5,012 | 4,946 | 4,272 | 4,939 | 0.987 | 0.852 | 0.985 | 1.156 |
| lmsys | 4,175 | 3,874 | 3,407 | 4,156 | 0.928 | 0.816 | 0.996 | 1.220 |
| mix | 6,729 | 6,925 | 5,318 | 7,142 | 1.029 | 0.790 | 1.061 | 1.343 |
| **기하평균** | | | | | **0.979** | **0.839** | **0.990** | **1.180** |

harvest 측 (MBM, 그룹 `harvest`):

| 셀 | mbm_total 평균 | llc_occ | 비고 |
|---|---:|---:|---|
| C1_sep20 | 16.5 GB/s | ~241 MB | MBA20 상한 동작 |
| C2_sib100 | **75.6 GB/s** | ~288 MB | 무제한 |
| C3_sib20 | 16.2 GB/s | ~214 MB | **실효 21.4% ← MBA 20% 설정** (캘리브레이션 1점) |

## 3. 판정

1. **연좌 throttle 미발생** — C3(sibling+MBA20) 가 C0 와 동률 (기하평균 0.990, ±noise).
   연좌가 있었다면 C3 < C2 여야 하나 실측 C3/C2 = **+18%**. 해석: vLLM 호스트 스레드는
   대부분 sleep (SUB_162: 96-100% S) 이고 순간 메모리 요청률이 낮아, 코어-레벨 max
   throttle 의 노출 시간이 미미. **D1 게이트 불충족 → 코어-배타 배치 강제 불필요.**
2. **무가드 sibling harvest 는 유해** — C2/C0 = 0.839 (−16%, humaneval 최악 −25%).
3. **MBA 가드만으로 충분** — CAT 없이 MBA 20% 만으로 −16% → −1% 회복.
   이 워크로드의 binding 간섭은 LLC 용량이 아니라 **메모리 BW** (llc_occ 288MB 점유
   상태에서도 C3 무해 — capacity 간섭 부차적).
4. **frontier**: (serving 99%, harvest 16 GB/s) ↔ (serving 84%, harvest 76 GB/s) —
   중간점들은 T1.5 MBA 캘리브레이션이 채움.

## 4. 방법론 발견 (v1 폐기 사유 — 이후 전 SUB 에 적용)

단일 부팅으로 셀을 이어 측정하면 **suffix global tree 가 셀을 넘어 누적 학습**
(`suffix_decoding_max_cached_requests=10000` default) — accept 0.72→0.86 단조 상승,
tps 가 최대 **+24% (mix)** 부풀어짐. v1 (단일 부팅) 데이터는 `runs_v1_singleboot/`
에 보존 (드리프트 정량 증거). **규칙: spec-decode 측정의 셀 비교는 반드시 셀별
fresh boot** (SUB_212 의 config-당-부팅과 동일 원칙).

## 5. 산출물

| 파일 | 내용 |
|---|---|
| `runs/summ_*.json` | v2 본판정 28셀 |
| `runs/mon_C{1,2,3}*.csv` | harvest MBM 시계열 (2s 간격) |
| `runs_v1_singleboot/` | v1 28셀 + 드리프트 증거 |
| `run_sub214.sh` | 재현 스크립트 (v2, per-cell boot) |

## 6. 후속 연결

- **SUB_215 (D2 L2 CAT SMT) GO** — 단, 본 결과로 설계 단순화: MBA20 가드가 이미
  sibling 간섭을 −1% 까지 막으므로, D2 의 질문은 "L2 CAT 추가가 *잔여* (humaneval
  −9% 류 worst-case) 를 더 줄이는가" 로 재정의.
- T1.5 (MBA 캘리브레이션) 에 본 측정의 (20% → 21.4%) 1점 기여.
- 논문 §9: 연좌 미발생 + BW-지배 간섭 + frontier 1쌍 — design rule 표의 1행.
