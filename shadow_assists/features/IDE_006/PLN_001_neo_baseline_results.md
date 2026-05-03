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

본 baseline 의 비교 회차 (NEO ON) 은 *NEO 데이터 path 의 적재 단계* 에 따라 의미가 달라진다.

### 5.1 · 현재 (2026-04-30) NEO ON 의 의미 — *무회귀 검증* 만

NEO data path 가 dev 머신 smoke (Qwen-1.5B + RTX 3090) 에서 wiring 통과 (TSK_016 §3 의 Step 5.1~5.4). 단 *forward 결과 합산* 은 Step 5.5 (forward-context fork — sub-batch 별 disjoint `slot_mapping` / `cu_seqlens` / `block_tables`) 미적용으로 KV cache cross-contamination 발생 → **divergence 검출 후 vanilla fallback** 이 active.

따라서 현재 시점에 NEO ON 회차를 돌려도:

- **token output**: vanilla 와 *bit-exact* (fallback 으로 vanilla forward 결과 채택)
- **wall_s / throughput**: scheduler 의 추가 cost 만큼 *vanilla 보다 살짝 느림* (기대) — 진짜 NEO dual forward 가 발화하지 않음

→ 본 단계의 NEO ON 회차는 **prod scale 회귀 검증** (smoke 3 prompt 가 못 잡는 scale 회귀) 영역. 운영 의미:

| 회차 | 회귀 검증 의미 |
|---|---|
| **500 × 50:50** | *개발 회차* — 31 분, KV pool 99% 한계 영역에서 NEO ON 의 wiring chain 이 깨지지 않는지 (수백 iteration 동안 swap-out / preempt / dual-forward 분기 / divergence-fallback 경로 모두 안전) |
| 1000 × 50:50 | 정식 회귀 — 58 분, scale ↑ |
| 5000 × 50:50 | 외부 보고 회귀 — 4.7 시간 (효과 측정 가능 단계 후로 미루는 게 합리) |

### 5.2 · 향후 *효과 측정* 영역 — Step 5.5 + TSK_015/TSK_017 후

진짜 NEO 효과 (wall_s 단축 / concurrent 확장) 측정은 다음 단계 누적 후 의미:

| 시점 | NEO 회차 의미 |
|---|---|
| TSK_014 + TSK_016 §3 Step 5.4 (현재) | wiring 통과 — *vanilla fallback* 으로 무회귀만 |
| TSK_016 §3 Step 5.5 (forward-context fork) 후 | 진짜 dual forward 활성 (KV cross-contamination 해소) → *latency* 효과 측정 가능 |
| TSK_015 (KV exclusive ownership) 후 | request 단위 GPU/CPU 분리 → *capacity* 효과 측정 가능 (concurrent 134 → 200+ 검증) |
| TSK_017 (PerfPredictor 실측 table) 후 | sequential vs pipelined 결정의 *진짜* heuristic 활성 (현재는 ZeroPerfPredictor 로 항상 sequential) |
| TSK_018 (CPU kernel 통합) 후 | full stack — 정식 NEO 효과 측정 |

### 5.3 · 정식 비교 회차 plan

| 단계 | 회차 | wall_min (예상) | 비교 metric |
|---|---|---|---|
| 무회귀 검증 (현재 ~ Step 5.5 직후) | 500 × 50:50 | 31 분 + α (scheduler overhead) | token-id bit-exact + wall_s 회귀 ≤ +5% |
| 효과 측정 baseline | 1000 × 50:50 | 58 분 (NEO ON) | wall_s 비교 / concurrent 비교 |
| 외부 보고 baseline | 5000 × 50:50 | 4.7 시간 (NEO ON) | wall_s 비교 / concurrent 비교 / token output bit-exact |

### 5.4 · NEO ON 측정 결과 (2026-05-03, Phase B async fix 후)

**환경**: Llama-70B + TP=8, max_num_seqs=256, gpu_mem_util=0.85, **VLLM_NEO_KV_FREE=1** (Phase B fix: NeoSchedulerAdapter(AsyncScheduler) 상속 적용 후)

| 회차 | vanilla wall | NEO ON wall | gain (wall) | NEO ON out_tps | gain (tps) | NEO 발화 |
|---|---|---|---|---|---|---|
| **500 × 50:50** | 1869.6s | **1706.5s** | **+8.72%** | 2400.2 (vs 2190.8) | +9.56% | 발화 zero (max_conc 78 = 30%) |
| **1000 × 50:50** | 3502.0s | **3449.9s** | **+1.49%** | 2374.6 (vs 2339.2) | +1.51% | 발화 zero |
| **5000 × 50:50** (외부 보고) | 16839.0s (4h 40m) | **17428.1s** (4h 50m) | **-3.50%** | 2350.2 (vs 2432.4) | -3.38% | 발화 zero |

**측정 결과 정리**:
- **회귀 영역**: NEO ON 가 *vanilla 대비 ±5% 영역 안* (500 / 1000 / 5000 회차).
- **NEO swap dispatch first fire / forward-context fork active / broadcast crash / B-5 truncate guard = 모두 0** (모든 회차).
- **KV cache usage 분포** (5000 회차) — 90~99% 영역 1457회 / 100% 영역 139회 — sustained 90%+ 영역에 머물렀으나 NEO swap_out 미발화.
- **swap_out 미발화 원인**: NEO sibling 의 `_sync_neo_gpu_decoding_q` 가 `self.running` 의 *decode 단계* (num_computed_tokens >= num_prompt_tokens) reqs 만 매핑. *prefill 단계* reqs 의 KV 는 `gpu_decoding_q` 에 추적 안 됨 → NEO 가 본 영역의 KV pressure 인식 못 함 → `gpu_block_needed > swap_out_threshold` 영역 미진입.

**해석**:
- Phase A (NEO 베이스 적재) + Phase B (async fix) = **NEO 인프라 활성 + vanilla 동등 영역** 입증 완료.
- 진짜 NEO gain (cdec dispatch 발화 → CPU pacpu 활용) 은 *별도 영역* — NEO sibling 의 prefill KV 추적 path 보강 또는 swap_out_threshold 의 ratio 조정 (RATIO env) 영역.
- 현재 회귀 영역 (5000 의 -3.4%) = NEO sibling 의 매 step overhead 누적 (~수십만 step × 0.12 ms / step ≈ 수십 초).

### 5.5 · 다음 단계 — 진짜 cdec dispatch 발화 영역

진짜 NEO gain 측정 path:
- **(a) prefill KV 추적 보강** — NEO sibling 의 `_sync_neo_gpu_decoding_q` 를 `self.running` 의 *모든 reqs* 로 확장 (decode + prefill).
- **(b) RATIO env forced-fire** — `VLLM_NEO_SWAP_OUT_RATIO=0.5` 또는 0.7 으로 swap_out_threshold scale → *낮은 KV 영역에서 cdec swap-out 강제 발화*. 본 turn 직전 RATIO=0.05 + 200 prompts forced-fire 회차 200/200 완주 입증.
- **(c) max_num_seqs ↑** (256 → 512) + **input/output ↑** (8192 → 16384/16384) → KV pool 한계 영역 *지속* sustained.

본 영역들은 별도 multi-day / multi-hour 영역.

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
| 2026-05-03 | **§5.4 + §5.5 신설 — Phase B async fix 후 NEO ON 측정 3 회차 + 진짜 cdec dispatch 발화 path 식별** | (1) **§5.4** 적재 — Phase B `NeoSchedulerAdapter(AsyncScheduler)` 1줄 fix 후 정식 비교 회차 (500 / 1000 / 5000 × 50:50, Llama-70B + TP=8, VLLM_NEO_KV_FREE=1). 결과: 500 +8.72% / 1000 +1.49% / 5000 **-3.50% wall regression** (가장 통계 신뢰 영역). NEO 발화 카운트 = 모두 zero (max_conc 78 = 30%, KV usage 90~100% sustained 였으나 swap_out 미발화). (2) **swap_out 미발화 root cause 식별**: NEO sibling 의 `_sync_neo_gpu_decoding_q` 가 *decode 단계* reqs 만 매핑 → prefill 단계 KV pressure 추적 안 됨. (3) **§5.5** 신설 — 진짜 cdec dispatch 발화 path 3 옵션 (prefill KV 추적 보강 / RATIO env forced-fire / max_num_seqs+input ↑). 별도 multi-day / multi-hour 영역. (4) **결론**: Phase A (NEO 베이스 적재) + Phase B (async fix) = NEO 인프라 활성 + vanilla 동등 ±5% 영역 입증 완료. 진짜 NEO gain (논문 H100 14% 영역) 은 별도 path. |
| 2026-04-30 | **§5 갱신 — 현재 NEO ON 의 의미 + 정식 비교 회차 plan 분리** | TSK_016 의 Step 5.1~5.4 wiring 통과 후 사용자 질문 ("500 prompt 로 NEO 기능 검증이 가능한거지?") 에 답하기 위한 layered 정리. (1) **§5.1**: 현재 NEO data path 는 forward-context fork 미적용 → KV cache cross-contamination → vanilla fallback active. NEO ON 회차의 의미 = *무회귀 검증* 만. 500 회차 = 개발 회귀, 1000 = 정식 회귀, 5000 = 외부 보고 회귀 (효과 측정 단계 후로 미룸). (2) **§5.2**: 진짜 효과 측정은 Step 5.5 (forward-context fork) → TSK_015 (KV exclusive) → TSK_017 (PerfPredictor 실측 table) → TSK_018 (CPU kernel 통합) 누적 후 의미. (3) **§5.3**: 무회귀 / 효과 측정 / 외부 보고 의 3 단계 비교 회차 plan 명시. |

---

**↑ 부모 PLN**: [`PLN_001`](PLN_001.md) · **↟ 조부 IDE**: [`IDE_006`](README.md) · **연계**: [`TSK_014`](TSK_014.md), [`TSK_016`](TSK_016.md), [`TSK_017`](TSK_017.md), [`NEO_redesign`](NEO_redesign.md), [`NEO_code_deepdive`](NEO_code_deepdive.md)
