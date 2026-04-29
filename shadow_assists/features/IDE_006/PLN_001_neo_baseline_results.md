**↑ 부모 PLN**: [`PLN_001`](PLN_001.md) · **↟ 조부 IDE**: [`IDE_006`](README.md) · **연계**: [`TSK_014`](TSK_014.md), [`TSK_016`](TSK_016.md), [`TSK_017`](TSK_017.md), [`NEO_redesign`](NEO_redesign.md), [`NEO_code_deepdive`](NEO_code_deepdive.md)

---

# PLN_001 deliverable — NEO baseline 측정 결과 (vanilla vLLM)

| 항목 | 값 |
|---|---|
| 측정 일자 | 2026-04-29 |
| 측정 환경 | Intel Xeon Platinum 8480+ x2 + NVIDIA H100 80GB HBM3 × 8 |
| vLLM 버전 | `0.1.dev15917+g0a6396b45` (editable, branch `feat/ide006-neo-asymmetric`) |
| 모델 | `meta-llama/Llama-3.3-70B-Instruct` (TP=8) + `Qwen/Qwen2.5-1.5B-Instruct` (TP=1) |
| script | `eval/run_neo_baseline.py` (+ wrapper `eval/run_neo_baseline.sh`) |
| dataset | `benchmarks/sonnet.txt` (Shakespeare line 임의 sampling) |
| NEO 활성 | **모두 OFF** (vanilla — `enable_neo_asymmetric=False`) |

> **목적**: NEO 식 architecture (4 차 재정의) 의 *진짜 효과 측정* 의 비교 baseline. 본 회차들은 *모두 vanilla* — NEO 의 data path 가 진짜 변경된 후 (TSK_014~018 적재 + 모델 forward 통합 + GPU model runner hook 까지 완료) 동일 시나리오의 NEO 회차와 비교.

---

## 1. 측정 시나리오 분류

| 영역 | 시나리오 | 의미 |
|---|---|---|
| **KV 충분** (NEO 효과 영역 *밖*) | 작은 input + 작은 batch | vanilla 무회귀 검증용 |
| **KV 한계** (NEO 가치 영역 *안*) | 큰 input × 큰 batch — concurrent 가 max_num_seqs 못 채움 | 진짜 NEO 효과 측정 baseline |
| **input-heavy** | 긴 prompt + 짧은 output | TSK_009 input_heavy_B 시나리오 재현 가능 영역 |

---

## 2. 6 회차 baseline 결과

### 2.1 · Qwen-1.5B 정확도 검증 회차 (e2e smoke)

| 항목 | 값 |
|---|---|
| 모델 / TP | Qwen2.5-1.5B-Instruct / 1 |
| num_prompts / input / output | 3 / ~10 / 16 |
| 의미 | enable_neo_asymmetric=True 활성 시 vanilla 와 *bit-exact* 동등 입증 |
| 결과 | token-id equality **PASS** + NeoSchedulerAdapter / execute_model gate / sub-batch attachment log 모두 PASS |
| 결과 file | `eval/run_neo_e2e_smoke.sh` 실행 시 `/tmp/neo_smoke.log` |

### 2.2 · Llama-3.3-70B + TP=8 — KV 충분 영역

| n | input | output | wall_s | prompt_tps | output_tps | req/s | concurrent | KV usage |
|---|---|---|---|---|---|---|---|---|
| **100** | 2048 | 16 | **9.9** | 21,535 | 161.6 | 10.10 | <100 | 17% |
| **1000** | 2048 | 16 | **88.7** | 24,032 | 180.3 | 11.27 | ~123 | 21% (queue 처리) |

→ KV pool 충분 영역. concurrent 가 *KV 한계 못 도달*. vLLM 의 *queue 관리* 로 throughput 안정적. **NEO 효과 영역 밖**.

### 2.3 · Llama-3.3-70B + TP=8 — input-heavy 영역

| n | input | output | wall_s | prompt_tps | output_tps | req/s | concurrent | KV usage |
|---|---|---|---|---|---|---|---|---|
| **100** | 15360 (avg 5418) | 1024 | **70.2** | 7,720 | 1,459 | 1.42 | ~100 | 한계 |

→ *target* input 15360 이지만 sonnet line 짧아 *avg 5418*. concurrent 100 (max_num_seqs 한계). 이전 TSK_009 input_heavy_B (165s) 대비 *2.4× 빠름* — vLLM 의 chunked prefill / continuous batching 향상.

### 2.4 · Llama-3.3-70B + TP=8 — KV 한계 영역 (진짜 NEO 가치 영역)

50:50 input/output (target 8192 / 8192) 시나리오 — concurrent 가 KV pool 한계로 256 max 못 채움.

| n | wall_s | wall_min | prompt_tps | output_tps | req/s | concurrent | KV usage |
|---|---|---|---|---|---|---|---|
| **500** | 1,870 | 31 | 1,449 | 2,191 | 0.267 | ~134 | 99% |
| **1000** | 3,502 | 58 | 1,547 | 2,339 | 0.286 | ~134 | 99% |
| **5000** | 16,839 | **281 (4:39:38)** | **1,609** | **2,432** | **0.297** | ~134 | 99.7% |

→ **진짜 NEO capacity 가치 영역**. KV pool 99.7% 한계 도달 + concurrent 134 / 256 = 52% (KV 부족으로 더 못 채움).

### 2.5 · 회차 별 적당함

| 용도 | 회차 | wall_min | 이유 |
|---|---|---|---|
| **개발 단계** (NEO data path 변경 후 빠른 회귀) | **500** | 31 | steady state 도달, 30 분 안에 완료 |
| **정식 NEO 효과 비교** | **1000** | 58 | 통계 안정성 충분 (5000 대비 차이 ~5%) |
| **외부 보고 / 논문** | **5000** | **281** | 가장 신뢰성. 사용자 결정 baseline |

3 회차의 *steady state throughput 차이* ~11% (5000 vs 500) — 이미 *KV 한계 + concurrent 134 + steady state* 도달.

---

## 3. 핵심 발견

### 3.1 · KV 한계 영역의 throughput 14× 저하

| 영역 | prompt_tps |
|---|---|
| KV 충분 (100/1000 × 2048) | 21,535 / 24,032 |
| **KV 한계 (500/1000/5000 × 50:50)** | **1,449 / 1,547 / 1,609** |

**14× 저하** — KV pool 제한이 throughput 의 *진짜* bottleneck. 이게 NEO 가 해소하려는 영역.

### 3.2 · concurrent 134 < max 256 (52%)

KV pool 1.27M tokens 한계로 vLLM scheduler 가 *concurrent 134 reqs 만* 처리. 256 까지 못 채움. NEO 의 *진짜 evict + sub-batch dual forward* 가 이 영역에서:
- **capacity 효과**: concurrent 256 까지 채워 *2× throughput 향상* 가능 영역
- **속도 효과**: GPU/CPU 동시 forward 로 *latency 향상* 가능 (단 NEO_code_deepdive §4 의 layer-offset 메커니즘 활성 후)

### 3.3 · NEO baseline 비교의 정식 영역

NEO 의 진짜 효과 측정은 **5000 × 50:50** baseline (4.7 시간) 와 *동일 시나리오 NEO 회차* 비교. NEO data path 활성 (TSK_014~018 + 모델 forward stage 분할 + GPU runner hook) 후 동일 회차 1 회 추가.

### 3.4 · *vanilla 무회귀* 가 우선

NEO data path 변경 시 *동일 input → 동일 output* 보장 (NEO_code_deepdive §1 invariant). 즉 NEO 회차의 *token output* 은 vanilla 와 bit-exact. throughput / capacity 만 향상.

본 baseline 의 *token output* 은 NEO 회차의 *correctness reference*.

---

## 4. 적용 결과 file

| 시나리오 | JSON | log |
|---|---|---|
| Qwen-1.5B e2e | (script 내장) | `/tmp/neo_smoke.log` |
| 100 × 2048 | `/tmp/neo_baseline_100.json` | `/tmp/neo_baseline_100.log` |
| 100 × 15360 input-heavy | `/tmp/neo_baseline_input_heavy.json` | `/tmp/neo_baseline_input_heavy.log` |
| 1000 × 2048 | `/tmp/neo_baseline_1000.json` | `/tmp/neo_baseline_1000.log` |
| 500 × 50:50 | `/tmp/neo_baseline_500_5050.json` | `/tmp/neo_baseline_500_5050.log` |
| 1000 × 50:50 | `/tmp/neo_baseline_1000_5050.json` | `/tmp/neo_baseline_1000_5050.log` |
| **5000 × 50:50 (논문 baseline)** | `/tmp/neo_baseline_5000_5050.json` | `/tmp/neo_baseline_5000_5050.log` |

> /tmp/ 위치는 dev 환경 임시. **정식 적재**: `eval/results/20260429_225037_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_neo_baseline/` (2026-04-29 archived).

---

## 5. 다음 단계 — NEO 회차 비교 측정 영역

본 baseline 의 비교 회차 (NEO ON) 은 다음 시점에 *진짜 의미*:

| 시점 | NEO 회차 의미 |
|---|---|
| TSK_014 (Scheduler 활성) 후 | scheduler 의 sub-batch 결정만 적용 — 데이터 path vanilla → 동일 결과 예상 |
| TSK_016 (Asymmetric pipelining) 후 | 진짜 dual forward 활성 → *throughput / capacity 측정* |
| TSK_018 (CPU kernel 통합) 후 | full stack — 정식 NEO 효과 측정 |

**5000 × 50:50 NEO 회차** = TSK_018 까지 완료 후 1 회 측정 — 4.7 시간. 그 결과를 본 baseline 과 비교하여:

- **wall_s 비교** — NEO 의 *throughput 향상 비율*
- **concurrent 비교** — NEO 의 *capacity 확장* 비율 (134 → 200+ 가능?)
- **token output bit-exact** — vanilla 무회귀 검증

---

## 6. References

- 부모 PLN: [`PLN_001`](PLN_001.md)
- 부모 IDE: [`IDE_006`](README.md), [`NEO_redesign`](NEO_redesign.md)
- 알고리즘 reference: [`NEO_code_deepdive`](NEO_code_deepdive.md) §3 (Scheduler), §4 (Asymmetric pipelining), §5 (BlockManager)
- 신규 TSK: [`TSK_014`](TSK_014.md), [`TSK_015`](TSK_015.md), [`TSK_016`](TSK_016.md), [`TSK_017`](TSK_017.md), [`TSK_018`](TSK_018.md)
- 측정 script: `eval/run_neo_baseline.py`, `eval/run_neo_baseline.sh`, `eval/run_neo_baseline_5050_chain.sh`
- 이전 TSK_009 reference: `eval/results/20260429_043734_*_tsk009_validation/` (input_heavy_B, output_heavy_B, equal_B)

---

## 7. Change Log

| 날짜 | 변경 | 사유 |
|---|---|---|
| 2026-04-29 | PLN_001 deliverable 신규 발행 (본 문서) | NEO 4 차 재정의의 vanilla baseline 측정 6 회차 적재. 사용자 결정 (5000 × 50:50 = 정식 논문 baseline). NEO 비교 회차의 reference. |

---

**↑ 부모 PLN**: [`PLN_001`](PLN_001.md) · **↟ 조부 IDE**: [`IDE_006`](README.md) · **연계**: [`TSK_014`](TSK_014.md), [`TSK_016`](TSK_016.md), [`TSK_017`](TSK_017.md), [`NEO_redesign`](NEO_redesign.md), [`NEO_code_deepdive`](NEO_code_deepdive.md)
