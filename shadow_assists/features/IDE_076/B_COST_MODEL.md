# B_COST_MODEL — 생산자 비용 기반 정적 Hot 배치의 비용표·선별 모델 (PLAN.md §8). 초안 2026-09-17 23:25 KST

## 1. 계약 (§8.3)
- 논리 expert (l,e), Hot 집합 H = 층별 물리 슬롯 0..N_l−1 의 logical id (hotmap `physical_to_logical_map`), Σ N_l = 5,952 (`layer_budget.per_layer`). 검사기: `eval/ide076/hotmap_tools.py validate` (순열·합·범위), `swap` (승격/강등 쌍·층 간 슬롯 이동), `diff`.
- H0: hotmap_v2 (7ce02cad…) + layer_budget_5952 (c8ba4247…), hot_sets_digest 68f74531…. 후보는 H0 파일을 덮어쓰지 않고 `placements/HB_<id>/` 에 저장하고 부팅 시 `--init-expert-location` + `KT_GPU_EXPERTS_PER_LAYER` 로 적용(§8.9).
- rank 복제: TP4 는 각 rank 가 같은 논리 Hot 집합의 FP8 expert shard 를 보유(23,808 = 4×5,952 로그 계수). HBM 검증은 부팅 시 rank 별 사용량(nvidia-smi·allocator) 으로.

## 2. 비용 정의 (§8.5~8.6)
- C_j(H): 생산 job j 의 실제 남은 Cold 서비스 span (NUMA 병렬 union). 측정 근거: replay n 스케일(A1_GAP_ANALYSIS §6) — **expert 당 ≈ 62 µs (R 무관, W 스트리밍 한계)**, job 고정비 ≈ 0. 서버 S29 도 expert 당 ≈ 52 µs 로 일치. 따라서 screening 은 `C_l ≈ c1 × E[unique cold experts in job]` (c1 = R_CPU 확정 후 재측정; 현재 v4/v5 62 µs).
- 승격 expert e 의 한계 절감: ΔC = c1 × P_job(e) (bs=63 job 에 e 가 등장할 확률) — 실제로는 'unique' 감소분이므로 근사. 강등의 증가 비용도 같은 식. 출처 표기 `measured slope × calibration freq`.
- G_j(H): GPU Hot 경로 비용. 층별 hot expert 수와 rows 분포에 따라 변함. **미측정 (estimated)** — 같은 층 교환은 hot 수 불변이라 ΔG≈0 로 두되, 층 간 이동은 ΔG 를 측정해야 함: 방법 = CORR 트레이스의 hot 커널 span(layer 별 hot_span p50, overlap_budget) 을 N_l 에 대해 회귀 (S29 데이터로 1차, 후보 부팅에서 실측 2차).
- 소비 지연 연결(§8.7 screening): 층 l 의 late_share·cold_pub_delta 가 클수록 ΔC 가 GPU 대기(다음 층 진행) 로 노출 → 우선순위 = late_share_l × ΔC. 기존 release 시각 고정 replay 는 순위용이며 tok/s 예측이 아님.

## 3. calibration 분리 (§8.2)
| 집합 | 정의 | 용도 |
|---|---|---|
| CALIBRATION | sonnet 행 조합 seed 20261001 / 20261002, 입력 512·출력 128, C64 n128 (`M076_QWEN_CALIB`, `_B`); 서버 `--expert-distribution-recorder-mode stat` + `/start|dump|stop_expert_distribution_record` → `logical_count[L,E]` (전 expert 빈도, GPU 포함) + v3 기록기 CORR (job 비용·expert 표본) | 비용표·후보 생성 |
| SELECTION | seed 20261011 (`M076_QWEN_SELECT`) OFF ×2 | 후보 실측 선별 |
| CONFIRMATION | 동결 HB, 새 부팅 MAIN_SHORT/C1/LONG (고정 입력) + 새 seed 독립 입력 1 종 | 최종 판정 |
MAIN_SHORT(seed 20260916 고정 manifest) 은 calibration 에 쓰지 않는다. 각 집합의 prompt hash·seed·tokenizer revision 을 CALIBRATION_MANIFEST.json 에 기록.

## 4. v0 결과 (2026-09-17 23:16, calibration = IDE_070 measure_baseline 빈도, 비용 = S29)
- 층별 기대 unique cold 2~6 vs S29 관측 6~16 → **빈도 세트 불일치** (H0 는 그 빈도에서 커버리지 98.6~99 % 이나 현재 워크로드는 cold 선택 ≥2.4 %). H0 의 층별 순서는 그 빈도 기준 최적이라 같은 층 교환은 v0 에서 순이득 음수. → §3 의 재수집이 선행 조건. 파일: `eval/results/IDE_076_20260917/calibration/v0_S29/` (참고용, 채택 근거 아님).

## 5. 후보 생성 규칙 (§8.8; R_CPU 고정 후)
1. 새 calibration 으로 층별 P_job(e) 계산, CORR 로 층별 late_share·cold_pub_delta·hot_span 갱신.
2. 같은 층 교환: cold 중 P_job 최대 ↔ hot 중 P_job 최소 (순이득 > 0 인 경우만).
3. 층 간 이동: CPU 가 이른 층(cold_pub_delta < 0, 예: S29 의 31/46/32) 의 최소 P_job hot 을 강등 → 늦은 층(late_share ≈ 1, cold_pub_delta 최대: 20/42/21/13) 의 최대 P_job cold 승격. 슬롯 합 불변, HBM 검증.
4. 후보당 map diff·비용 출처·coverage 를 `placement_candidate.jsonl` 에 기록, SELECTION 실측 → CONFIRMATION.

## 6. calibration v1 (2026-09-18 00:12, CALIB_R0_CALIB_000347: sonnet seed 20261001/20261002, recorder stat, CORR)
- recorder `logical_count[1000 steps, 62, 160]` → step 별 선택 수로 디코드(≤512/층)·prefill 청크 step 분리: 디코드 132 step, prefill 267 step.
- **H0 커버리지: prefill 분포 0.985(최소 0.979) vs 디코드 분포 0.951(최소 0.901).** H0(IDE_070) 는 prefill 이 지배하는 집계 빈도로 만들어져 디코드 라우팅에는 덜 맞음 → 디코드 기대 cold 선택/토큰 24.4 (prefill 7.3) → bs=63 job 당 기대 cold 선택 ≈ 25/층 → 관측 unique cold 12~18/서브풀과 정합 (v0 의 과소 추정 원인 해소).
- 따라서 B 의 1차 후보는 **디코드 빈도로 재선정한 Hot 집합(HB_DEC1)**: 같은 greedy knapsack(5,952 슬롯, 층별 빈도순) — `eval/ide076/hb_from_freq.py`. 승격/강등 수·층별 예산 변화·두 분포 커버리지는 placements/HB_DEC1/stats.json. 슬롯 수 동일 → HBM 동일(부팅 시 확인). 수치·품질 검증(§8.10) 후 SELECTION(seed 20261011) → CONFIRMATION(MAIN/C1/LONG).
- ΔG: 층 간 슬롯 이동이 크므로(층별 예산 변화) hot 커널 비용 변화를 CORR 트레이스 회귀(delta_g_regression) 와 후보 부팅 실측으로 확인.

## 7. calibration v1 비용표·ΔG·후보 (2026-09-18 01:28, `calib_build.py` → `calibration/v1/`)
- 입력: CAL1/CAL2 CORR v2 분석(층별 service·unique cold·cold_pub_delta·late_share·overlap_budget, bs=63 디코드 그래프 스텝만) + recorder 디코드 전용 빈도(132 step) + slope 62 µs/expert(replay n 스케일).
- **모델 검증**: 층별 기대 unique cold(모델) 평균 13.3 vs 관측 p50 평균 14.8 → v0 의 2~6 vs 6~16 불일치가 디코드 전용 빈도로 해소됨(§4 의 원인 = 빈도 세트).
- 소비 지연 연결: late_share > 0.9 인 층 = 4~23 (CPU cold 가 GPU 보다 늦게 끝나는 층이 연속 구간), cold_pub_delta < 0 (CPU 가 이른) 층 = 0, 1 뿐 → 층 간 이동의 '이른 층' 공급원이 거의 없음. 따라서 슬롯 재배분보다 **같은 슬롯 수에서 디코드 빈도 순으로 재선정(HB_DEC1/HB_MIX1)** 이 주 후보.
- ΔG 회귀(`delta_g_regression.json`, estimated): overlap_budget ≈ 1,358 µs + 1.6 µs × N_L, r² = 0.005 → hot 커널 비용의 hot 수 의존은 이 데이터에서 검출되지 않음(층별 예산 변화 ±수십 슬롯 × 1.6 µs ≪ cold 62 µs/expert). 후보 부팅 실측(SELECTION)으로 확인.
- 단일 교환 후보 12 개(`candidate_swaps.jsonl`, placements/HB_SW_*/HB_XL_*): 각 순 ΔC ≈ 61~62 µs/job (승격 expert P_job ≈ 1, 강등 ≈ 0) = 스텝(≈63 ms) 의 0.1 % → 판정 한도(2 %) 미만이라 **모델 단계에서 선별 제외**(placement_candidate.jsonl 에 기록). 
- 집계 후보 기대값(calibration 분포 내, 낙관적): 디코드 토큰당 기대 cold 선택 H0 24.4 → HB_DEC1 6.4 / HB_MIX1 9.3; prefill 커버리지 H0 0.985 → DEC1 0.923 / MIX1 0.981 (DEC1 은 prefill 비용 증가 위험 → LONG/C1 회귀 게이트로 확인). 실측 선별은 SELECTION(seed 20261011) 부팅 4 회(H0, DEC1, MIX1, H0) — post_fix 체인에 포함.

## 8. SELECTION 결과 (2026-09-18 08:48~09:15, v8f 플래그 OFF, seed 20261011, 배치당 2 부팅 × 2 세션)
| 배치 | tok/s 평균 (min~max) | H0 대비 | TTFT p50 | TPOT p50 |
|---|---|---|---|---|
| H0 | 645.2 (634.2~656.5) | 기준 | 2,242 | 73.7 |
| HB_DEC1 | 504.1 (500.0~508.0) | **−21.9 %** | 4,816 | 80.5 |
| HB_MIX1 | 682.3 (677.5~684.4) | **+5.7 %** | 2,271 | 67.0 |
- HB_DEC1(디코드 빈도만): prefill 커버리지 손실(0.985→0.923) 이 TTFT 2.1 배로 나타나 기각. 디코드 이득도 없음(TPOT +9 %, prefill 지연이 디코드 웨이브와 겹침).
- HB_MIX1(디코드+prefill 가중): TTFT 동등, TPOT −9 % → 두 부팅·네 세션 모두 H0 최대값(656.5) 을 넘음. **선별: HB_MIX1** → CONFIRMATION(동결 map, 새 부팅, MAIN×3/C1/LONG 고정 입력 + 새 seed 20261021 독립 입력 2 세션; H0 기준선 R0C 와 짝, B/A12B/A2B/A1B 각 3 블록) 은 post4 체인.
- NaN 사망 없음(6 부팅). 층별 예산·hotmap SHA 는 manifest 로 확인(HB_MIX1 0a415098…/71ec2db2…, HB_DEC1 28f7d8e3…/3ef39a76…).
