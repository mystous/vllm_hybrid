# Stage 3 — A5 + A1~A4 조합 매트릭스 측정 결과 (2026-05-21 KST)

> **출처**: 사용자 명시 (turn 15) — "턴 종료 하고 A5 진행 하고 A5 포함된 조합 테스트 진행해"
> **측정**: HEAD `0d7dc0334` (turn 14 후), 100p × 8192, gmu=0.85, env-ON baseline + A5 (FA3 Sequence-Aware Split Heuristic, `VLLM_NEO_MAX_NUM_SPLITS=8`) + A1~A4 조합 12 tests.
> **Stage 1 baseline (t00)**: 932.2 tps — 본 매트릭스의 비교 기준.
> **연관**: [Stage 1 RESULTS](../stage1_a1a2a3a4_matrix_100p_20260521/RESULTS.md), [N 문서](../../analysis/N_cdec_leftover_elimination_ideas.md) 영역 A.

---

## 1. 조합 매트릭스 (A1-A2 동일 OMP runtime, 동시 사용 X)

| # | Test | Levers |
|---|---|---|
| 12 | t12 | A5 단독 (max_num_splits=8) |
| 13 | t13 | A1+A5 |
| 14 | t14 | A2+A5 |
| 15 | t15 | A3+A5 |
| 16 | t16 | A4+A5 |
| 17 | t17 | A1+A3+A5 |
| 18 | t18 | A1+A4+A5 |
| 19 | t19 | A2+A3+A5 |
| 20 | t20 | A2+A4+A5 |
| 21 | t21 | A3+A4+A5 |
| 22 | t22 | A1+A3+A4+A5 |
| 23 | t23 | A2+A3+A4+A5 |

## 2. 측정 fact (12 tests, 시간순 — 각 ~15 min)

| Test | Levers | tps | wall (s) | crash | Δ vs Stage 1 baseline (932.2) |
|---|---|---:|---:|:-:|---:|
| t12 | A5 | 930.7 | 871.7 | 0 ✓ | -0.2% |
| t13 | A1+A5 | 916.8 | 884.7 | 0 ✓ | **-1.7% ⚠️** |
| t14 | A2+A5 | 931.5 | 871.0 | 0 ✓ | -0.08% (noise) |
| **★ t15** | **A3+A5** | **938.2** | **873.2** | 0 ✓ | **+0.6% ⭐ BEST** |
| t16 | A4+A5 | 921.6 | 880.1 | 0 ✓ | -1.1% |
| t17 | A1+A3+A5 | 931.8 | 879.2 | 0 ✓ | -0.04% (noise) |
| t18 | A1+A4+A5 | 932.6 | 878.4 | 0 ✓ | +0.04% (noise) |
| t19 | A2+A3+A5 | 934.1 | 877.0 | 0 ✓ | +0.2% |
| t20 | A2+A4+A5 | 921.6 | 880.2 | 0 ✓ | -1.1% |
| t21 | A3+A4+A5 | 931.0 | 871.3 | 0 ✓ | -0.1% |
| t22 | A1+A3+A4+A5 | 925.8 | 876.4 | 0 ✓ | -0.7% |
| t23 | A2+A3+A4+A5 | 919.9 | 881.8 | 0 ✓ | -1.3% |

## 3. lever 별 분석

### 3.1 A5 단독 (max_num_splits=8) — noise

- A5 단독 = 930.7 tps (-0.2% vs Stage 1 baseline)
- A5 = FA3 split heuristic cap. paper sweet spot=8 권고.
- 본 workload 에는 적용 효과 없음 — 800+ active task vs H100 132 SM 이미 포화 상태 → split cap 무의미.

### 3.2 best = A3+A5 (+0.6%) — 작은 synergy

- t15 = 938.2 tps (Stage 1 best A4 단독 941.1 vs Stage 3 best A3+A5 938.2 — 비슷)
- A3 단독 (Stage 1) = -1.1% 손실이었으나, **A5 와 결합 시 +0.6% 로 회복**
- 추정: FA3 split cap 으로 KV access pattern 이 더 sequential 해지면서 prefetch hint 가 살아남 — 단 ±2% noise 안 (3-run 검증 필요)

### 3.3 anti-synergy = A1+A5 (-1.7% ⚠️)

- t13 = 916.8 tps — Stage 3 최악 (Stage 1 best A4 단독 941.1 대비 -24.3 tps)
- Stage 1 A1 단독 (+0.4%) vs Stage 3 A1+A5 (-1.7%) — A5 가 A1 효과를 reverse
- libomp thread placement vs FA3 split heuristic 충돌 (Stage 1 의 A1+A4 anti-synergy 와 동일 패턴)

### 3.4 A4 + A5 = -1.1% (Stage 1 A4 단독 +1.0% 가 사라짐)

- t16 (A4+A5) = 921.6 vs t04 (A4) = 941.1 → Δ = -19.5 tps (-2.1%)
- A4 단독의 +1.0% win 이 A5 결합 시 완전히 cancel out
- FA3 split heuristic 이 GPU SM 활용을 바꾸면서 NUMA-local malloc 효과가 무력화

## 4. 핵심 발견

### 4.1 Stage 3 winner (A3+A5 +0.6%) < Stage 1 winner (A4 단독 +1.0%)

- A5 가 단독 best 를 만들지 못함 — **A5 자체는 무효**
- FA3 split heuristic 의 가정이 본 workload (대량 active task) 에 맞지 않음 (decode 가 낮은 head 일 때만 유효)

### 4.2 모든 측정 ±2% 안 (single 1-run noise)

- Stage 1 baseline 932.2 ± 2.0% = 913.6 ~ 950.8
- Stage 3 의 모든 12 측정 (916.8 ~ 938.2) 이 본 구간 안
- → **statistical significance 없음** (3-run avg 로 winner 재검증 필요)

### 4.3 stability (crash)

- 12 tests 모두 crash = 0 ✓
- A5 (FA3 split heuristic env override) 의 본 환경 안정성 확인

### 4.4 Stage 1 + Stage 3 결론

- **모든 A-tier lever (A1-A5) 의 효과가 ±2% noise 안** 으로 수렴
- 유일 win = A4 단독 (+1.0%, numactl --localalloc)
- **A-tier (Quick win / low risk) 소진** → B-tier (FlashDecoding++, OmniServe LSE async) 로 이행 권고

## 5. 다음 turn 권고

| 우선순위 | 작업 | effort | 이유 |
|---|---|:-:|---|
| **★★★** | **B-tier 진입** (B1 FlashDecoding++ unified-max softmax, B2 OmniServe LSE async merge) | 1-2 일 | A-tier 소진 (모두 ±2% noise) |
| ★★ | A4 단독 3-run avg (statistical confidence) | 90 min | Stage 1+3 유일 win 검증 |
| ★ | A1+A5, A4+A5 anti-synergy 원인 분석 (별도 micro-profile) | 30 min | 다음 lever 설계 인사이트 |
| ⚪ | A1/A2/A3/A5 lever 영구 폐기 (winner 외 제외) | — | 후속 sweep 단순화 |

## 6. raw 측정 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260521_180457_stage3_a5_matrix_100p/SUMMARY.tsv` |
| per-test 결과 (12 dirs) | `eval/results/20260521_180457_stage3_a5_matrix_100p/t12~t23/` |
| launch script | `/tmp/run_stage3_a5_matrix.sh` |
| code 변경 | `vllm/v1/attention/backends/flash_attn.py` (A5 max_num_splits env override) |
| 사전 install | Intel libomp (`/workspace/vllm_dev_prj/lib/libiomp5.so`, Stage 1 에서 install) |

## 7. Stage 1 + Stage 3 합산 (24 measurements)

| Stage | 측정 수 | best | worst | net |
|---|---|---|---|---|
| Stage 1 (A1~A4) | 12 | A4 단독 941.1 (1-run) → **noise** (SUB_032 avg 930.2) | A1+A3 920.9 (-1.2%) | A4 마저 noise |
| Stage 3 (A5+) | 12 | A3+A5 938.2 (+0.6%) → noise band | A1+A5 916.8 (-1.7%) | A5 무효 |
| **합산** | **24** | ~~A4 단독 941.1~~ → 모두 noise | A1+A5 916.8 (-1.7%) | **A-tier 전체 무효, B-tier 만이 유효 path** |
