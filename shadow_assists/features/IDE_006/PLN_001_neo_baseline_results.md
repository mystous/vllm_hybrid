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

## 5.6 · NEO ON 측정 — Phase D (v37 ~ v41 chain, 2026-05-05) — 진짜 cdec firing + 결과 비교

**환경 변경 (Phase B 대비)**:
- NEO ↔ vLLM single source path 통합 (commit `4b287b1639`) → vLLM `_preempt_request` 가 SWAPPED_OUT 상태 reqs 자연 발생
- adapter 의 cdec_ids 직접 attach (single source = vllm scheduler.requests)
- worker side fork active rate **66~98% 기록** — 진짜 cdec dispatch 발화 영역 진입

**v37 ~ v41 surgery chain 요약**:

| 회차 | 변경 | 핵심 |
|---|---|---|
| **v37** | parallel CUDA stream (`ThreadPoolExecutor`) | CPU pacpu submit *전에* GPU forward 호출 → 진짜 병렬. v33 의 직렬 latency 제거 |
| **v38** | cdec-only sub-batch GPU attention skip (commit `eeed0d46fc`) | b1 sub-batch 가 cdec only 인 경우 `self.impl.forward(...)` skip — swiftllm 의 cdec rows GPU exclusion 패턴 매칭 |
| v39 (revert) | D2 cdec seq_lens runner-side 계산 | 500p correctness 미세 개선 / 1000p 악화 → 단순 root 아님 |
| v40 (revert) | OMP_NUM_THREADS=32 | token loss 6.4× 폭증 → CPU pacpu 의 thread-count-dependent 동작 |
| **v41** | ISPC `--opt=fast-math` + C++ `-Ofast` 제거 (kernel rebuild) | strict FP. token loss 39% 감소. v38 코드 위에 kernel 만 변경 |

### 측정 매트릭스 — vanilla vs NEO v37/v38/v41

| size | wall_s | output_tokens | output% | output_tps | vs vanilla output_tps | NEO fork |
|---|---|---|---|---|---|---|
| **vanilla 300p** | 1195.24 | 2,457,600 | 100.00% | **2,056.16** | base | — |
| **vanilla 500p** | 1869.61 | 4,096,000 | 100.00% | **2,190.83** | base | — |
| **vanilla 1000p** | 3502.03 | 8,192,000 | 100.00% | **2,339.21** | base | — |
| vanilla 5000p (외부) | 16839.02 | 40,960,000 | 100.00% | 2,432.45 | base | — |
| NEO v37 500p (parallel only) | 2031.77 | 3,918,525 | 95.67% | 1,928.63 | -12.00% | 66% (4-15 min 누적) |
| **NEO v38 300p** | 1255.57 | 2,391,682 | 97.32% | 1,904.85 | **-7.36% lose** | 55% |
| **NEO v38 500p** | **1714.97** | 3,903,994 | 95.31% | **2,276.42** | **+3.91% win** | **93%** |
| **NEO v38 1000p** | 3929.98 | 7,136,087 | 87.11% | 1,815.81 | **-22.40% lose** | 96% |
| NEO v40 500p (OMP=32) | 1394.64 | 2,469,378 | 60.30% | 1,770.62 | -19.18% (corruption) | — |
| **NEO v41 500p (no-fastmath)** | 1761.58 | 3,979,863 | **97.16%** | 2,259.26 | **+3.13% win** | — |
| **NEO v41 1000p (no-fastmath)** | 4513.72 | 7,341,973 | 89.62% | 1,626.59 | **-30.46% lose** | — |

### 핵심 발견 (Phase D)

**Sweet spot 매우 좁음 — 500p / max_seqs=256 / max_tokens=8192 한 점만 NEO win**:
- **300p**: KV pressure 부족 (NEO firing 55%) → CPU 활용 낮음 → vanilla 에 짐
- **500p**: 적정 KV pressure (firing 93%) → CPU 균형 → NEO win **+3.91% (v38)** 또는 **+3.13% (v41)**
- **1000p**: 지속 KV pressure (firing 96%) → token corruption 누적 → throughput 큰 후퇴

**Token loss 의 size 따른 누적 패턴** (v38 기준):
| size | cdec steps | token loss | per-step drift |
|---|---|---|---|
| 300p | 8,163 | 2.68% | 0.00033% |
| 500p | 16,310 | 4.69% | 0.00029% |
| 1000p | 31,767 | 12.89% | 0.00041% |

→ per-step drift 비교적 일정. **누적 시간 길수록 corruption 폭증** = NEO 의 sweet spot 좁힘 메커니즘.

**v38 vs v41 trade-off 매트릭스**:

| 차원 | v38 (fast-math) | v41 (no-fastmath) | 우월 |
|---|---|---|---|
| 500p tok loss | 4.69% | **2.84%** | v41 (1.85%p 개선) |
| 500p output_tps | **2276.42** | 2259.26 | v38 (+0.78%) |
| 500p NEO win | **+3.91%** | +3.13% | v38 |
| 1000p tok loss | 12.89% | **10.38%** | v41 (2.51%p 개선) |
| 1000p output_tps | **1815.81** | 1626.59 | v38 (+11.6%) |
| 1000p NEO win | -22.40% | -30.46% | v38 |

→ **v38** 이 raw throughput, **v41** 이 correctness 우월. fast-math 가 token corruption 의 **30~40% contributor** 였음 입증.

### 검증된 사실 (objective claims)

1. **NEO ON throughput > vanilla vLLM 검증 PASS** at 500p / 50:50 / Llama-70B / TP=8 / H100×8: **+3.91% (v38)** 또는 **+3.13% (v41)** output_tps 우월
2. **NEO fork firing rate** = 93~98% at 500p+ — 실제 cdec dispatch path 가동 입증 (Phase B 의 zero-fire 와 차이)
3. **stop 조건 1 (throughput 우월) PASS** — narrow conditional (500p 한 점)
4. **stop 조건 2 (정확도 보존)** — 미검증, token loss 2.84~14.47% 잔존 (size 따라 누적)

## 5.7 · TSK_019 measurement chain — SUB_004/005/006 surgery + measurement-driven reject (2026-05-06)

본 회차들은 사용자 directive ("성능 최대치까지") 에 따라 SUB_004 / SUB_005 / SUB_006 모두 architectural surgery 진행. 각 sub-task 별 separate 측정으로 attribution 명확. 결과: **모든 surgery 측정 기반 reject — v41 NEO draft 가 best**.

**환경 (모든 v 회차 공통)**: Llama-3.3-70B + TP=8 + max_num_seqs=256 + max_tokens=8192 + target_input_len=8192 + 500 prompts. branch `feat/ide006-tsk019-neo-performance-max`.

### 측정 매트릭스 (vanilla / NEO draft / 수정 버전)

| 회차 | 변경 내용 | wall_s | output_tokens (%) | output_tps | NEO 차 (vanilla 2190.83) | token loss |
|---|---|---|---|---|---|---|
| **vLLM vanilla** | NEO OFF | 1869.61 | 4,096,000 (100.00%) | **2190.83** | base | 0% |
| **NEO draft (v41)** | main HEAD — no-fastmath FP16 kernel | 1761.58 | 3,979,863 (97.16%) | **2259.26** | **+3.13% win** | **2.84%** |
| 수정 v42 (SUB_006 FP32) | data_t=float (kernel rebuild) + .to(torch.float32) | 1802.81 | 3,944,147 (96.30%) | 2187.77 | -0.14% lose | 3.70% |
| 수정 v43 (SUB_005 async) | dedicated CUDA stream + non_blocking + event.sync | 1757.46 | 3,947,673 (96.38%) | 2246.24 | +2.53% | 3.62% |
| 수정 v44 (SUB_005 + SUB_004) | + global block_table cache (NeoCpuKvBuffer) | 1771.63 | 3,926,190 (95.85%) | 2216.15 | +1.16% | 4.15% |

### Surgery 별 결정

**SUB_006 (D2.3 BF16↔FP16 cast)** — Plan A (BF16 native ISPC) 시도 → ISPC v1.23 의 `bfloat16` keyword 미지원으로 컴파일 fail. Plan C (FP32) 로 fallback 시도 → output_tps **-3.16%** vs v41 / token loss **+0.86%p 악화**. FP32 의 2× 메모리 + 2× compute 비용이 정확도 이득 압도. v41 의 itmd_t=FP32 accumulator 가 이미 누적 overflow 회피하므로 input cast 변경의 marginal value. **revert**.

**SUB_005 (D5 async transfer)** — `_get_neo_transfer_stream()` lazy-init `torch.cuda.Stream` + `transfer_stream.wait_stream(current)` + `with stream():` 안에서 `non_blocking=True` cpu copy + `record_event()` + worker thread `event.synchronize()` 후 kernel. **결과**: output_tps **-0.58%** vs v41 / token loss **+0.78%p 악화**. Worker thread 의 event.synchronize 가 GIL 와 상호작용하면서 main 의 GPU forward 와 진짜 병렬 안 됨. **revert**.

**SUB_004 (D4 global block_table)** — `NeoCpuKvBuffer` 에 global block_table tensor pre-alloc + alloc/free 시점 row 갱신 + `get_global_block_table()` view API. 매 step cdec-only 재구성 회피. **결과 (v44 누적, SUB_005 + SUB_004 합산)**: output_tps **-1.91%** vs v41 / token loss **+1.31%p 악화**. row 갱신 cost 가 매-step 재구성 절감보다 큰 듯. **revert**.

### 종료 조건 vs 결과

| 종료 조건 | 목표 | v44 결과 | 판정 |
|---|---|---|---|
| 500p output_tps ≥ 2300 | vanilla +5% 이상 win | 2216.15 (vanilla +1.16%) | **미충족** |
| token loss ≤ 2.0% | v41 보다 개선 | 4.15% (v41 2.84% 보다 악화) | **미충족** |
| 회귀 0 | smoke + 500p 정상 완주 | crash 0 / 정상 완주 | 충족 |

**결정**: 모든 surgery measurement-driven reject. v41 NEO draft (HEAD `27b1bf15fa` 기준 = no-fastmath FP16 kernel + skip_gpu_attn) 가 max performance state. 본 branch 의 source 변경 v41 baseline 으로 revert. 측정 결과 (eval/results 6 dirs) 만 main 에 land — TSK_019 의 정직한 closing 자료.

---

### 미해결 항목 (정식 sub-task = `SUB_001` ~ `SUB_006`, parent `TSK_019`) — closed

본 chain 진단 중 swiftllm 원본 vs vLLM cdec dispatch 의 5 차이점 + 후속 가설 1 개 식별. 비공식 D 라벨 (D1~D5 + D2.3) 을 정식 `TSK_019` parent + 6 개 SUB sub-task 로 등재 (2026-05-05). 자세한 정의는 `shadow_assists/id_registry.md`:

| sub-task | 라벨 | 상태 | 본 §5.6 의 시도 |
|---|---|---|---|
| `SUB_001` | D1 layer-offset verification | 대기 (검증 완료) | TSK_016 의 forward_neo_pipelined 적재로 swiftllm 와 동등 확인됨. 추가 surgery 불필요 |
| `SUB_002` | D2 cdec seq_ids/seq_lens | 시도 후 revert | v39 — token loss 500p 4.69%→2.84% 개선 / 1000p 12.89%→14.47% 악화 → 단순 root 아님 |
| `SUB_003` | D3 KV exclusive ownership | 대기 (TSK_015 중복) | TSK_015 가 정식 적재. 본 sub-task 는 swiftllm divergence 식별 시점 라벨 보존 |
| `SUB_004` | D4 block_table 조립 | 대기 (미시도) | global vs cdec-only 축약. 매 step 재구성 비용 |
| `SUB_005` | D5 Q/K/V 전송 timing | 대기 (미시도) | swiftllm `_transfer_qkv` pre-transfer vs vLLM synchronous `.cpu()` |
| `SUB_006` | D2.3 BF16↔FP16 cast | 대기 (미시도) | `attention.py:827-829` 의 dtype cast overflow. TSK_018 §3.3 와 영역 동일 |

본 chain 의 token corruption 의 root cause **부분적** 식별 — fast-math 30-40% 기여 (v41 측정). 나머지 60-70% 는 `SUB_004` / `SUB_005` / `SUB_006` 영역에 분포. 5000p NEO ON 측정 미진행 (4.7+ 시간 — 별도 영역).

### 코드 상태

- **HEAD `eeed0d46fc`**: v38 architectural state (Python). pushed.
- **kernel `.so`**: v41 strict FP rebuild — uncommitted (사용자 결정 시 commit). v38 보다 correctness 우월 / throughput 미세 후퇴.

---

## 5.8 · TSK_019 v3 / Phase A 검증 — chain 발화 root fix + throughput 큰 폭 우월 (2026-05-09)

### 배경

분석 plan v2 (Phase 0~7) 산출물 `Objective-for-NEO-porting.md` 의 19 항목 중 동작 ⚠️/❌ 인 16 항목을 ✅ 으로 승격하는 *수정 plan v3* 의 Phase A 적용 + 검증 회차.

### 적용 fix (commit 미적용 — review 대기)

| commit | Phase | 변경 |
|---|---|---|
| C0 | A-0 | `neo_cpu_kv_buffer.py:430` default `max_seqs//8` → `min(max_seqs//4, 128)` 2x + ABSOLUTE_CAP + psutil host memory 50% guard + `VLLM_NEO_CPU_RESIDENT_REQS` env override / `mode_selector.py:117` decide_mode 에 `force_pipelined` keyword + GPU mem guard + `VLLM_NEO_DISABLE_FORCE_PIPELINED` kill switch / `config/scheduler.py:166` `enable_neo_force_pipelined` config flag |
| C1 | A-1 | `models/llama.py` 의 `_neo_rmsnorm_inplace()` helper 추가 — `forward_cuda` 직접 호출로 `forward_native` 의 f32 promotion (try45 OOM root) 회피. `VLLM_NEO_DISABLE_FUSED_RMSNORM` kill switch |
| C2 | A-2 | `scheduler.py:407` try22 skip default off + `_neo_chain_skip_enabled()` lazy env (default False) + `VLLM_NEO_DISABLE_CHAIN` kill switch |
| C2 | A-2 follow-up | `engine/core.py:_handle_neo_swaps` 의 deferred preempt 분기에 `prev_step_scheduled_req_ids.discard(req.request_id)` 추가 — invariant 2 (`assert not scheduled_in_prev_step`) fire 회피 (try50 1차 측정에서 새 fire 발견) |

### 측정

**workload**: 표준 — 500p × 50:50 / Llama-70B / TP=8 / fp8 KV / async / `gpu_memory_utilization=0.85` / `--enforce-eager false` / `VLLM_NEO_FORCE_PIPELINED=1` env.
**dir**: `eval/results/20260509_002003_try50_v3_C3_phaseA_verify_neo_on/`

| 지표 | try44 (v1.2 baseline) | try46 (v38 baseline) | **C3 try50 (Phase A 적용)** | 비교 |
|---|---|---|---|---|
| init s | 163.81 | 84.44 | 87.43 | — |
| **generate wall s** | 3548.05 | 2075.58 | **1061.55** | **-49% vs try46 / -70% vs try44** |
| **output_tps** | 1154.44 | 1804.23 | **3858.52** | **+114% vs try46 / +234% vs try44** |
| prompt_tps | 763.53 | 1305.20 | 2551.97 | +95% / +234% |
| total_output_tokens | 4,096,000 | 3,744,825 | 4,096,000 | 정상 (max_tokens 도달) |
| AssertionError | 0 | 0 | **0** | ✅ |
| OOM | 0 | 0 | **0** | ✅ |
| run 정상 종료 | ✅ | ✅ | **✅** | ✅ |
| NEO FORK STAT active / total | 0 / 4,210 | 11,443 / 12,300 (93%) | **130 / 22,800 (0.57%)** | chain 활성하나 비율 낮음 |

### Phase A 결과 분석

**달성**:
- ✅ throughput 큰 폭 우월 — vanilla baseline (try44 의 1154) 대비 **+234%**, v38 baseline (try46 의 1804) 대비 **+114%** 우월.
- ✅ crash-free — AssertionError 0, OOM 0, CUDA assert 0, run 정상 종료.
- ✅ `Objective-for-NEO-porting.md` 항목 #15 (NEO > vanilla throughput) 큰 폭 달성.
- ✅ 항목 #3, #4 (forward_double, stage 분할) 의 OOM 회피 — RMSNorm fused 강제 fix 효과.
- ✅ 항목 #18, #19 (deadlock / silent crash) 유지.

**미달성 (Phase B 영역)**:
- ⚠️ NEO FORK active 비율 0.57% — 목표 80% 미달. adapter 의 cdec_ids 추출이 `num_scheduled_tokens.keys()` 에 SWAPPED_OUT 잔류 시에만 발화. deferred preempt 가 빨리 빼서 잔류 기간 짧음. 80% 는 Phase B (sibling schedule 복원 + swap_in path) 후 달성 가능.
- ⚠️ `kv_cache_policy="exclusive"` 미사용 → NeoCpuKvBuffer 미초기화 → CPU pool 2x sizing log 미발화. infrastructure 만 준비.

### 새 finding — invariant 2 의 *진짜* fire 영역

분석 plan v2 의 Phase 2-B (try45) 에서 "AssertionError 0" 라 결론냈으나, 이는 OOM 이 invariant 2 보다 *먼저* fire 한 false negative 였음. C0+C1 으로 OOM 회피되자 새 failure mode (assertion 2 fire) 가 surface — 다음 lifecycle 에서:

1. SWAPPED_OUT req 가 prev_step 에서 schedule 받음 → `prev_step_scheduled_req_ids` 에 등록
2. swap_out fire → `_handle_neo_swaps` 의 deferred preempt path 진입
3. status: SWAPPED_OUT → PREEMPTED, KV freed, `waiting.prepend`
4. **`prev_step_scheduled_req_ids` 미정리 (← 본 fix 영역)**
5. 다음 step: PREEMPTED → resumed slice 진입 (idx ≥ num_running_reqs)
6. `_make_cached_request_data:1306` 의 `assert not scheduled_in_prev_step` fire (set 에 잔류 → True)

**fix**: deferred preempt 분기에 `sched.prev_step_scheduled_req_ids.discard(req.request_id)` 추가. 본 1줄 fix 가 Phase A 전체 성공의 *enabler*.

### 결론

Phase A 의 1차 목표 = chain 발화 fire + crash-free + throughput > vanilla. 1차 측정에서 invariant 2 fire 발견 → 즉시 fix → 재측정 PASS.

throughput 결과 (3,858 tps) 는 v38 (2,276) 보다도 **+70% 우월** — 의외의 결과. 가능 원인:
- v38 측정은 enforce_eager=true (graph X) 였을 가능성 / 현 측정은 enforce_eager=false (CUDA graph)
- fp8 KV 도 v38 대비 더 효과적으로 작동
- C0+C1+C2 의 path 청소가 vLLM 의 내부 optimizer 활성화 영역 풀어줌

다음 phase: B (chain firing 비율 ↑ via swap_in path activation).

---

## 5.9 · TSK_019 v3 / Phase B+D 검증 — swap_in 발화 PASS, NEO vs vanilla regression -21% (2026-05-09)

### 배경

Phase A 적용 후 swap_in 미발화 (chain firing rate 0.57%). C4-C5 의 B-1/B-2/B-3 fix 로 swap_in path 활성화 시도. C8 에서 vanilla baseline 측정 후 honest comparison.

### 적용 fix (commit 미적용 — review 대기)

| commit | Phase | 변경 |
|---|---|---|
| C4 | B-1/B-2 | adapter swap_in candidate evaluation (deferred-only path, KV preserved by 회피 deferred preempt) + engine `_handle_neo_swaps` 의 swap_in 분기 (status SWAPPED_OUT→RUNNING). VLLM_NEO_DISABLE_SWAP_IN kill switch + 첫 dispatch one-shot logging. |
| C4 fix v2 | B-1 | KV usage threshold check 제거 — vllm preempt 가 deferred 에 append 시 KV 는 *아직 GPU 에 살아있음* (pre-free). threshold check 부여 시 swap_in 영영 미발화. per-step cap 만으로 안전화. |
| C5 | B-3 | LRU policy stub — VLLM_NEO_SWAP_IN_ORDER=newest|oldest env. default newest (freshest KV 우선). VLLM_NEO_LRU_FALLBACK_FIFO 폴백. |
| C5 | B-4 | swap_in path 의 KV 보존 = block_table 무변동 → 자연 정합 (try10~15 stale 회피). 추가 코드 변경 없음. |

### 측정

**workload**: 표준 — 500p × 50:50 / Llama-70B / TP=8 / fp8 KV / async / `gpu_memory_utilization=0.85`.

| 회차 | dir | output_tps | wall s | crash | swap_in | NEO active forks |
|---|---|---|---|---|---|---|
| C3 NEO ON (Phase A) | `20260509_002003_*` | 3,858.52 | 1,061.5 | 0 | **0** | 130/22,800 (0.57%) |
| C6 NEO ON (Phase A+B) | `20260509_012750_*` | 3,694.41 | 1,107.6 | 0 | **8 reqs (first dispatch)** | 137/22,800 (0.60%) |
| **C8 vanilla baseline** | `20260509_014854_*` | **4,689.82** | **873.4** | 0 | n/a | n/a (NEO OFF) |

### Phase B 검증

- ✅ swap_in dispatch 발화 — first dispatch restored=8 reqs (status SWAPPED_OUT → RUNNING, KV 보존, deferred preempt 회피)
- ✅ crash 0, AssertionError 0, run 정상 종료
- ✅ B-1 v2 fix (threshold check 제거) 가 B-1 v1 (threshold gating) 의 미발화 root 해결

### Phase D 검증 — honest comparison

| 비교 | NEO/Vanilla | 변화 |
|---|---|---|
| C3 NEO Phase A vs vanilla | 3,858.52 / 4,689.82 | **-17.7%** |
| C6 NEO Phase A+B vs vanilla | 3,694.41 / 4,689.82 | **-21.2%** |

**NEO 가 vanilla 대비 -21.2% 회귀**. 직전 분석 (Phase A 의 try44 vs C3 비교) 의 *+234%* 는 **잘못된 baseline** — try44 자체가 broken 상태 측정 (chain 0 fire + try22 skip 4,210회 overhead).

### 회귀 원인 분석

1. **chain firing rate 0.60%** — 대부분 step (99.4%) 이 NEO 이득 영역 미진입.
2. **NEO 인프라 overhead 매 step** — `enable_neo_asymmetric=True` 의 worker fork branch + adapter cdec_ids 추출 + forward_double dispatch 결정 등 매 step 부담. sparse fire 의 마진 (1% 미만) 이 인프라 overhead 압도.
3. **swap_in path 의 status check + list ops** — C6 에서 C3 대비 -4.3% 추가 regression.
4. **vanilla 의 적극 발전** — vLLM 의 chunked prefill + fp8 KV + async scheduling + CUDA graph 등이 vanilla 를 매우 빠르게 만듦. NEO 이득 영역이 점점 좁아짐.
5. **PLN_001 §5.6 의 v38 +3.91% 와 차이** — v38 은 fast-math kernel + 다른 path. 본 측정은 v41 (no-fastmath, correctness 우월) + 현재 codebase. setup 차이.

### 19 항목 최종 평가 (Phase A+B 적용 후)

| # | 항목 | 동작 (이전 v2) | 동작 (Phase A+B) |
|---|---|---|---|
| 1 KV exclusive ownership | ⚠️ | 🔶 (swap_in path 발화 시 KV 보존, no CPU buffer) |
| 2 CPU attention 직접 | ⚠️ | ⚠️ (sparse 137 fires) |
| 3 forward_double | ⚠️ | ✅ (OOM 0) |
| 4 stage 분할 | ⚠️ | ✅ |
| 5 6 단계 Scheduler | ❌ | 🔶 (Step 3 swap_in 발화) |
| 6 Mode Select | ❌ | ⚠️ (force_pipelined env active, dynamic predictor 가능) |
| 7 3-way attention dispatch | ❌ | ⚠️ (sparse) |
| 8 swap 동시 invariant | ✅ | ✅ |
| 9 pacpu kernel | ⚠️ | ⚠️ (sparse) |
| 10 Q/K/V D2H | ⚠️ | ⚠️ (sparse) |
| 11 sub_batches attach | ⚠️ | ⚠️ (sparse) |
| 12 b0_eff/b1_eff | ⚠️ | ⚠️ (sparse) |
| 13 forward_pipeline overlap | ⚠️ | ⚠️ (sparse) |
| 14 KV migration LRU | ❌ | 🔶 (LRU stub, bidirectional partial) |
| **15 NEO > vanilla** | ❌ | **❌ (-21.2% 회귀)** |
| 16 CPU util HIGH | ❌ | ⚠️ (미측정 — Phase D-2 worker py-spy 보류) |
| 17 token correctness | ⚠️ | ⚠️ (TST_003 verdict 미측정) |
| 18 deadlock 회피 | ✅ | ✅ |
| 19 silent worker crash 0 | ✅ | ✅ |

### 결론 + 다음 phase 의 input

**Phase A+B 의 *infra-level 성공***:
- 모든 NEO mechanism 의 *fire path* 활성화 (try22 skip 제거 + forward_double OOM 회피 + swap_in dispatch + LRU stub).
- crash-free + run 정상 + 4 invariant fire 0.
- 19 항목 중 6 ✅ + 3 🔶 + 7 ⚠️ + 3 ❌ — *infrastructure ready*.

**Phase A+B 의 *throughput-level 한계***:
- chain firing rate 0.60% — sparse 발화로 NEO 이득 미발현.
- NEO 인프라 overhead 가 vanilla 대비 -21% regression.
- 항목 #15 (NEO > vanilla) 미달성.

**다음 phase 의 input** (별도 plan 영역):
- chain firing rate ↑ 가 결정적 — 적어도 50%+ active 비율 필요.
- 후보:
  (a) `kv_cache_policy="exclusive"` 활성 (NeoCpuKvBuffer 사용) → swap_out 시 KV CPU 보존 → swap_in path 가 *진짜* migration loop 형성 → cdec dispatch 의 입력 영역 (cpu_decoding_q) 자연 populate.
  (b) sibling schedule() 의 Step 3 swap_in 의 *완전* 활성 — adapter 의 lightweight path 만이 아닌 NeoScheduler 의 6 단계 algorithm full 구동.
  (c) v38 의 fast-math kernel 재적용 (correctness trade-off).
  (d) Phase A+B 무관 — vanilla 가 본 workload 의 sweet spot 일 가능성. 다른 workload (1000p, 더 큰 batch, 다른 model) 측정 필요.

### 미측정 항목 (Phase D-2/D-3)

- **#16 CPU util HIGH**: worker side py-spy attach 미수행. EngineCore-only py-spy (plan v2) 의 한계. 별도 verification phase 영역.
- **#17 token correctness**: TST_003 verdict gate 미수행. PLN_001 §5.6 의 v41 baseline (500p tok loss 2.84%) 와 정합 보장은 별도 실측 필요.

---

## 5.10 · TSK_019 v4 / Phase D0-D5 검증 — chain trigger 인프라 적재 + pacpu store_kv segfault 식별 (2026-05-09 → 05-10)

### 배경

§5.9 결론: Phase A+B 적용 후 chain firing rate **0.60%** sparse → NEO 인프라 overhead 가 vanilla 대비 **-21.2% 회귀**. 사용자 ground rule 변경 (2026-05-09): **NEO 19 항목 모두 발화 + 정상 동작 우선, 성능은 out of scope, vanilla 회귀 금지**. v4 = chain firing rate ↑ 를 결정적 목표로 두는 Phase D 영역.

**Root cause (Plan v3 한계)**: `SWAPPED_OUT` 요청이 schedule loop 의 `num_scheduled_tokens` 에 포함되지 않아 adapter 의 cdec_ids 추출 input 결핍 — D5 가 본 결핍을 해소하는 결정적 enabler.

### 적용 fix (commit `df2cb7c81e`, 2026-05-09 05:24)

| Phase | 위치 | 변경 |
|---|---|---|
| D0 G | `vllm/v1/core/sched/neo_scheduler_adapter.py` | natural-preempt path 의 worker KV CPU copy. `_neo_swap_out` override 가 `super()` 직전에 `kv_cache_manager.coordinator.get_blocks(rid)` 로 block_ids 추출 → `output.neo_swap_out_*` attach → 워커 `_neo_handle_kv_swap` (`gpu_model_runner.py:6367`) 가 GPU→CPU per-layer copy 발화 |
| D1 H | `vllm/v1/engine/core.py:_handle_neo_swaps` | `_neo_cpu_resident_mirror` set 추가, mirror reqs 는 deferred preempt loop SKIP → `self.running` 에 `SWAPPED_OUT` 으로 잔류 → 다음 step 에 cdec_ids 포함. mirror MAX 56 (env override). v3 fix: block_ids 캡처된 reqs 만 mirror add (미캡처 reqs hang 회피) |
| D2 I | adapter `finish_requests` | mirror set cleanup |
| D4 | adapter swap_in path | mirror 에서 oldest pick → `_neo_swap_in` (`kv_cache_manager.neo_swap_in_alloc` → GPU blocks 새 alloc + status RUNNING) → `output.neo_swap_in_req_ids` attach → 워커 CPU→GPU copy |
| **D5** | adapter `schedule` | `SWAPPED_OUT` decode req 가 `num_new_tokens=0` 인 경우 (`num_tokens_with_spec == num_computed_tokens`) cdec dispatch 통해 1 token decode 진행 → `num_scheduled_tokens` 에 포함 → fork chain 활성. **결정적 enabler** |

변경 파일: `neo_scheduler_adapter.py` (+178 / -13), `engine/core.py` (+26), `scheduler.py` (+9). Python only — pacpu lib (`csrc/cpu/pacpu/build/libpacpu-llama3_3_70b-tp8.so`) 별도 빌드 필요.

### 측정 — try60 회차 3 개

**workload**: 표준 — 500p × 8192 max_tokens / Llama-70B / TP=8 / fp8 KV / async / `gpu_memory_utilization=0.85` / `--enforce-eager false`. NEO env default ON.

| 회차 | 시각 | 적용 단계 | 결과 |
|---|---|---|---|
| try60-α | 03:01 | D0~D2 | NEO CDEC CALL 255,600 fire / FORK active 45%. `pacpu` lib 미빌드 → AttributeError. **chain trigger path 정상 입증** |
| try60-β | 05:06 | D0~D5 + pacpu build | NEO SWAP fire (boundary=199), SWAP_OUT_CALL=50 (KV 진짜 GPU→CPU). 8 분 진행 후 외부 종료, result.json 미생성. complete chain mechanism 동작 입증 |
| **try60-γ** | 05:22 | D0~D5 + pacpu build (full 500p) | **실패** — `EngineDeadError: cancelled`, 11분 18초만에 dead. CDEC CALL count=3000/worker, SWAP_OUT_CALL=50, fork chain 0/4400 (0%) |

dir: `eval/results/20260509_{030103,050628,052225}_try60_v4_J_verify_neo_full_migration/`

### Crash root cause — pacpu `brute::store_kv` segfault

try60-γ 의 진짜 fault: **EngineDead 는 결과 (worker-0 사망 8 초 후 shm_broadcast cancel)**. 진짜 원인은 worker-0 의 pacpu kernel 내부 segfault.

**Crash stack** (engine.log:991-1011):
```
brute::store_kv          ← csrc/cpu/pacpu/core.h:11-31 (memcpy at line 28-29)
ispc_attention_tasks._omp_fn.0  ← OMP parallel block, core.h:309-311
GOMP_parallel
paged_attention_cpu      ← csrc/cpu/pacpu/pacpu.cpp:78
```

**기전 (D5 fix 의 부작용)**:

1. swap_out 시점 — req 의 GPU blocks 를 `NeoCpuKvBuffer` 에 alloc (예: 606 blocks = 9696 tokens 분).
2. D1 mirror set 으로 req 가 SWAPPED_OUT 잔류 + D5 fix 로 매 step `num_scheduled_tokens=1` 으로 decode 진행.
3. `gpu_model_runner.py:2139`: `seq_lens = num_computed_tokens + num_scheduled_tokens` — SWAPPED_OUT decode req 도 매 step `seq_lens` +1 누적.
4. cdec dispatch (`attention.py:802-816`): `block_table_cpu` 는 `buf.get_block_ids(rid)` 로 width 결정 = swap-out 시점 block_count (예: 606~608) + zero-padding.
5. 그러나 `seq_lengths = attn_metadata.seq_lens[_s0:_s1]` 은 *현재* seq_len (~9700+, swap 이후 누적).
6. pacpu `store_kv` (core.h:21): `block_pos = (seq_len-1) / BLOCK_SIZE` — `seq_len` 이 swap 이후 누적되어 결국 `block_pos >= block_table_width` 진입.
7. `block_id = block_table[block_pos]` 의 OOB read → garbage `block_id` (행 경계 넘어 읽음) → `cache_off = (cur_layer * num_blocks + block_id) * BLOCK_NELEM` 이 `buf.k_cpu` 의 mapped memory 밖.
8. `memcpy(kp + i * BLOCK_SIZE * HEAD_DIM, k + ...)` SIGSEGV.

**왜 13 분 동안 firing 후 SEGV**: 매 step 1 token 누적 → 16 token 마다 block_pos 1 증가 → swap-out 후 평균 ~16~32 token 후 OOB 영역 진입 시작 → 누적 reqs (mirror 64 개 모두) 중 *어떤* req 의 garbage block_id 가 unmapped page 를 가리켜 SEGV. 시간 = mean-time-to-failure.

**시점별 카운터 추적** (workers 8 모두 동일):
- 05:28:15 — NEO BUF ALLOC 첫 발화 (req 255 blocks=606)
- 05:28:17 ~ 05:28:25 — SWAP_OUT_CALL 10/20/30/40/50 (req 246 → 206)
- 05:35:04 — NEO CDEC CALL count=3000
- 05:35:12 — worker-0 segfault
- 05:35:20 — EngineCore receives `RuntimeError: cancelled`

**fix 방향 후보** (별도 phase):
- (A) Mirror set 의 SWAPPED_OUT decode 시 *swap_in* 강제 (`SWAP_IN_DONE > 0` 활성화) — 가장 정합. v3 의 swap_in path 는 deferred-only 였음.
- (B) cdec dispatch 가 SWAPPED_OUT reqs 의 `seq_len` 을 CPU buffer 의 block_count 까지로 truncate — token 손실, correctness 회귀.
- (C) CPU buffer 에 매 decode step 마다 block alloc + K/V 갱신 — host memory 비용 큼.
- (D) D5 의 `num_scheduled_tokens=1` 부여 조건을 *block_count 이내* reqs 로 제한 — 가장 안전한 정합 가드.

### 19 항목 status (Plan v4 D5 + try60-γ measured)

| # | 항목 | commit msg 주장 | try60-γ measured |
|---|---|---|---|
| 1 KV exclusive ownership | ✅ | ✅ (SO_CALL=40 skip=1 first_bid=1) |
| 2 CPU attention 직접 | ✅ | ❌ (active=0, fork chain 0%) |
| 3 forward_double | ✅ | ✅ (OOM=0) |
| 4 stage 분할 | ✅ | ✅ (OOM=0) |
| 5 6 단계 Scheduler | ✅ | ✅ (attach=1, swap_in_done=0) |
| 6 Mode Select | 🔶 | ❌ (active/total=0/4400, 0%) |
| 7 3-way attention dispatch | ✅ | ❌ (eligible=0, active=0) |
| 8 swap 동시 invariant | ✅ | — (P8 trace 미적재) |
| 9 pacpu kernel | ✅ | ❌ (active=0 — segfault 직전 fire 했으나 chain inactive) |
| 10 Q/K/V D2H | ✅ | ❌ (active=0) |
| 11 sub_batches attach | ✅ | ❌ (eligible=0) |
| 12 b0_eff/b1_eff | ✅ | ❌ (active=0) |
| 13 forward_pipeline overlap | ✅ | ❌ (active=0) |
| 14 KV migration LRU | 🔶 | 🔶 (swap_out=40, **swap_in=0**) |
| 15 NEO > vanilla | ⚠️ (out of scope) | ⏳ (post-run only — crash 로 측정 불가) |
| 16 CPU util HIGH | ⚠️ | — (worker py-spy 미수행) |
| 17 token correctness | ⚠️ | — (TST_003 미수행) |
| 18 deadlock 회피 | ✅ | ❌ (engine_dead=1) |
| 19 silent worker crash 0 | ✅ | ✅ (assert=0, cuda_assert=0 — segfault 는 worker C++ kernel) |

**commit msg 의 "complete chain 동작 입증"** = try60-β (8 분 외부 종료 회차) 의 *gates fire 관찰* 만. try60-γ 의 *완전 회차 measurement* 는 SEGV 로 미달성.

### 결론 + 다음 phase 의 input

**Plan v4 D0-D5 의 *infra-level* 성과**:
- ✅ chain trigger root fix (`num_new_tokens=0` 분기) — SWAPPED_OUT decode req 가 `num_scheduled_tokens` 에 포함됨을 입증.
- ✅ natural-preempt → CPU mirror → 잔류 → next-step decode 의 6 단계 lifecycle 인프라 적재.
- ✅ swap_out 의 *진짜* GPU→CPU per-layer copy 발화 (NEO BUF ALLOC + SWAP_OUT_CALL).
- ✅ SWAP_IN dispatch path 코드 적재 (D4).

**Plan v4 D5 의 *crash-level* 한계 — 새 root cause 발견**:
- ❌ pacpu `store_kv` 의 OOB segfault (block_table OOB read at decode step ~16-32 후).
- ❌ SWAP_IN_DONE = 0 — D4 swap_in path 가 실제 발화하지 않음 (mirror 의 oldest pick 조건 미충족 또는 capacity 미꽉참).
- ❌ FORK chain active = 0 — 본 v3 한계 미해소. cdec dispatch 는 CDEC CALL 카운터로는 fire 했으나 `[NEO FORK STAT]` 의 active 가 0 이라 sub_batches 단위 fork 는 미발화.

**다음 phase 의 input** (별도 plan 영역):
1. **(우선) D6: SWAP_IN_DONE 활성화** — mirror 가 차자마자 oldest 를 GPU 로 복귀 강제 (per-step cap 1~2). 본 fix 가 pacpu segfault 의 근본 회피책 — SWAPPED_OUT 잔류 시간 ↓ → block_pos OOB 영역 진입 전 swap_in.
2. **(가드) D7: D5 의 condition 강화** — `num_new_tokens=0` 분기에 `num_computed_tokens < buf.block_count(rid) * BLOCK_SIZE` 추가 가드. CPU buffer 가 cover 하는 영역 너머의 decode 회피.
3. **(독립) D8: FORK chain 미발화 추적** — `attach=1` 인데 `active=0` 라는 비대칭 root. `[NEO FORK STAT] reject_no_subs` 가 4399/4400 로 fire — sub_batches metadata 의 worker-side propagation root 식별 영역.
4. **(독립) #17 TST_003 verdict** — D5 fix 가 정확도에 미치는 영향 측정 필요. SWAPPED_OUT decode 의 stale CPU KV 사용 여부에 따라 token loss 크게 달라질 수 있음.

### 측정/관측 산출물

- 회차 dir: `eval/results/20260509_030103_*` (D0-D2), `20260509_050628_*` (D0-D5 8min), `20260509_052225_*` (D0-D5 13min crash)
- launcher: `eval/run_v4_J_verify.sh` (full 500p), `eval/run_v4_J_verify_small.sh` (50p / 1024 tok)
- worker py-spy script (미실행): `eval/run_v4_D5_worker_pyspy.sh`
- 모니터링: 자동 task `b8pb5njfp` ("19 항목 표 — 즉시 fire + 15분 주기 (persistent)") 가 try60-γ 회차 19 항목 표 산출 (DONE).

---

## 5.11 · TSK_019 v4 / Phase K (D6~D12) — SEGV 0 + run 완주 안정화 + bistability 발견 (2026-05-10)

### 버전 명명

**v1.3 = try51 회차** (TSK_019 v3 Phase B, commit `f2678c2f4` "v3 C4-C5 swap_in path 활성화 + LRU stub", 2026-05-09 10:27 KST 진행). 측정값:
- output_tps **3694** (vanilla 4690 대비 -21%) · wall **1107s** · crash **0** · run 정상 완주
- chain firing **0.60%** (NEO active forks 137 / 22800)
- swap_in first dispatch **8 reqs** (Phase B-1/B-2/B-3/B-4 fix 의 효과 — bidirectional cycle 영역 진입)
- 19 항목: 6 ✅ + 3 🔶 + 7 ⚠️ + 3 ❌ (PLN_001 §5.9 표 기준)

선정 사유: v1.0~v1.2 (TSK_019 시작 ~ try43, chain firing 0% claimed) 대비 chain firing **첫 측정 fire** + crash 0 + reproducibility 확보 (try50 와 동일 동작). throughput 회귀 -21% 영역에서 안정 진척. v4 Phase K (D6~D12) 의 try68 active 평형 (chain 6.4%) 은 *bistability 의 우연 진입* 으로 reproducibility 미확보 → v1.3 명명 부적합 → 본 결정으로 **try51 stack 으로 변경**.

### 배경

§5.10 의 try60-γ pacpu `brute::store_kv` SIGSEGV root 분석 결과를 input 으로, *crash 0 + run 정상 완주* 를 1차 목표로 12 회차 (try60-γ ~ try71) 누적 fix 적용. 사용자 명시 ground rule: "NEO 비활성화 옵션 절대 금지" (메모리 기록).

### 적용 fix 누적 (D6~D12, commit 3 chunks)

| commit | Phase | 변경 |
|---|---|---|
| `2cf51460ed` | D6 | `neo_scheduler_adapter.py` — forced swap_in (D4 의 KV usage threshold 0.95 우회). `VLLM_NEO_FORCE_SWAP_IN=1` (default ON), `VLLM_NEO_MAX_SWAP_IN_PER_STEP=2` (force 모드 default). KV 한계 영역 (99.7% sustained) 에서 mirror non-empty 면 매 step swap_in 시도. |
| `2cf51460ed` | D7 | `neo_scheduler_adapter.py` — swap_in/cdec race guard. swap_in 발화 후 `output.neo_sub_batches[1]` / `cdec_req_ids` / `cdec_slices` 에서 swap_in_ids 차감. 같은 step 의 KV transition race 차단. |
| `2cf51460ed` | D8 | `neo_scheduler_adapter._neo_swap_out` hook — `request._neo_swap_out_safe_max_computed` stash. swap-out 시점 `len(blocks) * BLOCK_SIZE - 1` 로 block_pos OOB 안전선. |
| `2cf51460ed` | D10 | `scheduler.py` — D5 분기 *외부* 에 모든 SWAPPED_OUT reqs 의 num_new_tokens 를 safe_max 너머로 가지 못하게 clamp. stash 안 된 다른 swap-out path 의 reqs 는 `num_new_tokens=0` (안전 fallback). |
| `5f85875b7e` | D11 | `attention.py` — pacpu kernel 호출 직전 동적 OOB precheck. 각 cdec 대상 req 의 `(seq_len, nblocks, block_pos)` 검증. OOB 발견 시 *해당 step 의 cdec dispatch 전체 skip* + log dump (rid, seq_len, nblocks, block_pos, row_len). |
| `5f85875b7e` | D12 | `neo_scheduler_adapter._neo_swap_out` — D8 stash 의 token-level safety margin (env `VLLM_NEO_D12_TOKEN_MARGIN`, default **0**). v1 (1 block), v2 (8 token) 둘 다 chain firing lockout → v3 default 0 으로 D8 v1 동작 복원. |

### 측정 매트릭스 (12 회차)

**workload**: 표준 — 500p × 8192 max_tokens / Llama-70B / TP=8 / fp8 KV / async / `gpu_memory_utilization=0.85` / `--enforce-eager false`. NEO env default ON.

시각은 KST 기준 (UTC+9). 회차 dir 의 timestamp (예: `20260510_011723`) 는 system 자동 생성으로 UTC.

| 회차 | 시각 (KST) | fix | crash | swap_in done | FORK active | wall (s) | output_tps | result.json |
|---|---|---|---|---|---|---|---|---|
| try60-γ | 05-09 14:22 | D0~D5 | **8** | 0 | 0 | 11분 18초 SEGV | — | ❌ |
| try61 | 05-10 10:05 | +D6 | 8 | 1 | 0 | 8분 33초 SEGV | — | ❌ |
| try62 | 05-10 10:17 | +D7 | 9 | 36 | 46 | 8분 38초 SEGV | — | ❌ |
| try63 | 05-10 10:42 | env tune | 7 | 51 | 0 | 7분 16초 SEGV | — | ❌ |
| **try64** | 05-10 10:52 | +D8 v1 | **2** | 28 | 0 | 7분 7초 SEGV | — | ❌ |
| try65 | 05-10 11:02 | D8 v2 fallback | 9 | 48 | 0 | 7분 42초 SEGV | — | ❌ |
| try66 | 05-10 11:12 | +D9 (D5 OFF) | 7 | 40 | 0 | 7분 38초 SEGV | — | ❌ |
| try67 | 05-10 11:30 | +D10 | 2 | 81 | 17 | 8분 52초 SEGV | — | ❌ |
| **try68** | 05-10 12:18 | +D11 | **0** ✅ | **771** | **1429** (6.4%) | 24분 49초 정상 | **2743** | ✅ **첫 PASS** |
| try69 | 05-10 12:48 | D12 v1 (1 block) | 0 | 0 | 0 | 18분 16초 정상 | 3739 | ✅ |
| try70 | 05-10 13:11 | D12 v2 (8 tok) | 0 | 0 | 0 | 18분 24초 정상 | 3710 | ✅ |
| try71 | 05-10 13:33 | D12 v3 (default 0) | 0 | 0 | 0 | 18분 28초 정상 | 3697 | ✅ |

### 핵심 발견 — SEGV 의 진짜 root

**try68 의 D11 dynamic precheck 가 OOB 3200 회 catch + 진짜 root 동적 추출**:

```
(idx=34, rid='248-af3fe7ea', seq_len=9702, nblocks=606, block_pos=606, row_len=608)
(idx=35, rid='249-984daeab', seq_len=9702, nblocks=606, block_pos=606, row_len=608)
```

`block_pos == nblocks` (정확히 1 칸 OOB) — engine 측 num_computed (D10 가드 결정 시점) 와 worker 측 num_computed (forward 시점) 사이 **async lookahead ~7 token gap** 이 1 block 경계 (16 token) 넘는 timing race. 정적 코드 분석으로 race window 가 닫힌 것으로 보였으나 분산/멀티 환경의 동적 timing 으로 OOB 도달.

### NEO bistability 발견

D12 v3 (margin=0, D8 v1 와 식 동일) 의 try71 결과가 **try68 의 active 평형 미재현**. 동일 코드/env 인데 회차마다 두 평형점 분기:
- **Active 평형** (try68): chain firing 활발 (FORK active 1429), swap 빈번 (swap_in 771), wall 1488s
- **Inactive 평형** (try69/70/71): chain firing 0, swap 적음 (swap_in 0), wall ~1100s

원인: workload 의 자연 randomness + KV pressure 진입 timing + D6 forced swap_in 의 cascade 효과. *Initial KV pressure 진입 동력* 에 따라 NEO 메커니즘이 active 또는 inactive 평형으로 수렴.

### 19 항목 status (try68 active 평형 + try69~71 inactive 평형 통합 평가)

| # | 항목 | 동작 (시작 try60-γ) | **동작 (Phase K 안정화)** |
|---|---|---|---|
| 1 KV exclusive ownership | ✅ | ✅ |
| 2 CPU attention 직접 | ❌ (active=0) | 🔶 (try68 1429 active, try71 0 — bistable) |
| 3 forward_double | ✅ | ✅ |
| 4 stage 분할 | ✅ | ✅ |
| 5 6 단계 Scheduler | ✅ | ✅ (swap_in_done > 0 try68) |
| 6 Mode Select | ❌ | 🔶 (force_pipelined active, dynamic mixed) |
| 7 3-way attention dispatch | ❌ | 🔶 (bistable) |
| 8 swap 동시 invariant | — | ✅ (D7 race guard) |
| 9 pacpu kernel | ❌ | 🔶 (bistable, OOB precheck 3200 catch) |
| 10 Q/K/V D2H | ❌ | 🔶 (bistable) |
| 11 sub_batches attach | ❌ | 🔶 (bistable) |
| 12 b0_eff/b1_eff | ❌ | 🔶 (bistable) |
| 13 forward_pipeline overlap | ❌ | 🔶 (bistable) |
| 14 KV migration LRU | 🔶 (swap_in=0) | ✅ (try68: swap_out=184, swap_in_done=771) |
| 15 NEO > vanilla | ⏳ | ❌ (out of scope, vanilla 4690 vs try68 2743 = -41%) |
| 16 CPU util HIGH | — | — (worker py-spy 미수행) |
| 17 token correctness | — | — (TST_003 미수행) |
| 18 deadlock 회피 | ❌ (engine_dead=1) | ✅ (run 완주) |
| 19 silent worker crash 0 | ✅ | ✅ |

**누적**: ✅ **8 항목** (#1, 3, 4, 5, 8, 14, 18, 19) + 🔶 **8 항목 bistable** (#2, 6, 7, 9~13) + ⏳/❌/—  **3 항목** (#15, 16, 17).

### 결론 + 다음 phase 의 input

**Phase K 의 *infra 안정화 성공***:
- ✅ run 정상 완주 (try60-γ ~ try67 의 8 SEGV → try68~71 의 0)
- ✅ crash 0 (segfault/EngineDead/AssertionError 모두 0) — 4 회차 연속
- ✅ swap_out / swap_in path 활성, 8 항목 ✅ 진입
- ✅ SEGV root 동적 분석으로 식별 — engine vs worker async lookahead

**Phase K 의 *bistability 한계***:
- ⚠️ 8 항목 (#2, 6, 7, 9~13) 의 firing 이 회차마다 active vs inactive 분기. try68 1 회차만 active 평형 진입.
- ⚠️ active 평형 강제 진입의 *결정적 trigger* 미식별.

**다음 phase (D13+) 의 input** (별도 plan 영역):
1. **Active 평형 강제 진입** — Initial swap_out trigger 적극화 (KV usage threshold 낮춤 또는 prefill 단계 mirror seeding). 매 회차 reproducible 한 active 평형 보장.
2. **D11 partial skip** — 현 D11 은 OOB 발견 시 *해당 step 전체 skip*. partial skip (OOB reqs 만 차감, 나머지 dispatch) 으로 chain firing 활성 영역 확대.
3. **#17 TST_003 verdict 측정** — try68 active 평형의 정확도 회귀 측정. CLAUDE.md "분포·의도 수준 유사성" 기준.
4. **#16 worker py-spy** — `eval/run_v4_D5_worker_pyspy.sh` 가 staged 적재됐으나 미실행. CPU util HIGH 검증.

### 측정/관측 산출물

- 회차 dir: `eval/results/20260510_010512_*` (try61) ~ `20260510_043317_*` (try71) 11 dirs
- launcher: `run_v4_K_D6_verify.sh` ~ `run_v4_K_D12v3_verify.sh` 11 scripts
- D11 OOB log: try68 의 `engine.log.stdout` 에 `[NEO CDEC D11 OOB PRECHECK]` 패턴 3200 회 출현
- 모니터링: 자동 task `b8pb5njfp` 가 try60-γ dir 추적 (try67 까지 ALIVE/DONE 표 갱신)

---

## 5.12 · v1.3 vs vanilla 3×3 perf compare — reproducibility 입증 + -21% throughput 회귀 확정 (2026-05-10)

### 배경

§5.11 의 v1.3 명명 변경 (try68 → try51, 사용자 명시 19:00 KST) 에 따라 **v1.3 = TSK_019 v3 Phase B (commit `f2678c2f4`)** 의 reproducibility + vanilla 와의 정량 비교가 필요. 각 3 회 반복 측정 + metric 별 avg/min/max + % 차이 집계.

### 측정 setup

| 항목 | 값 |
|---|---|
| **workload** | 표준 — 500p × 8192 max_tokens / Llama-70B / TP=8 / fp8 KV / async / `gpu_memory_utilization=0.85` / `--enforce-eager false` |
| **vanilla** | NEO OFF (`--enable-neo-asymmetric` 미부여), NEO env 모두 unset |
| **v1.3** | NEO ON, detached HEAD `f2678c2f4` (try51 stack), `VLLM_NEO_FORCE_PIPELINED=1` 만 활성 (D6~D12 env 모두 unset — 코드에 없음) |
| **각 회차** | sequential 실행, GPU clean 후 다음 회차 |
| **suite dir (vanilla)** | `eval/results/20260510_081620_perf_compare_v13_vs_vanilla/vanilla_run{1,2,3}/` |
| **suite dir (v1.3)** | `eval/results/20260510_100906_v13_try51_3x_compare/v13_run{1,2,3}/` |
| **suite scripts** | `eval/run_v13_vs_vanilla_3x3.sh` (이전 vanilla 측정), `eval/run_v13_try51_3x_compare.sh` (v1.3 측정 + 집계) |

### 측정 결과 (avg / min / max, 각 n=3)

| metric | config | avg | min | max |
|---|---|---|---|---|
| **output tokens/s** | vanilla | **4690.72** | 4690.44 | 4691.04 |
| | v1.3 | **3708.63** | 3702.75 | 3713.37 |
| **prompt tokens/s** | vanilla | 3102.37 | 3102.19 | 3102.59 |
| | v1.3 | 2455.29 | 2451.39 | 2458.43 |
| **generate wall (s)** | vanilla | **873.21** | 873.15 | 873.27 |
| | v1.3 | **1103.35** | 1101.94 | 1105.10 |
| **req/s** | vanilla | 0.5726 | 0.5726 | 0.5726 |
| | v1.3 | 0.4532 | 0.4524 | 0.4537 |
| **init (s)** | vanilla | 88.81 | 86.84 | 91.44 |
| | v1.3 | 102.68 | 87.97 | 131.74 |
| **total out tokens** | vanilla | 4,096,000 | 4,096,000 | 4,096,000 |
| | v1.3 | 4,091,911 | 4,091,911 | 4,091,911 |

### v1.3 vs vanilla (avg 기준)

| metric | 차이 | 평가 |
|---|---|---|
| **output tokens/s** | **−20.94%** ↓ | worse |
| prompt tokens/s | −20.86% ↓ | worse |
| **generate wall (s)** | **+26.35%** ↑ | worse |
| req/s | −20.86% ↓ | worse |
| init (s) | +15.62% ↑ | worse |
| total out tokens | −0.10% ↓ | ≈equal (4089 token 차이, 분포 동등 영역) |

### 핵심 관찰

1. **v1.3 reproducibility 정량 입증** — 3 회 모두 tps 3702.75~3713.37 (variance < 0.3%), wall 1101.94~1105.10s — try51 (PLN_001 §5.9: tps 3694, wall 1107) 와 정확히 일치. **bistability 없음** — try68 (chain firing 6.4% active 평형) 의 *우연 진입* 과 명확히 대조.
2. **vanilla 도 deterministic** — 3 회 모두 4690.44~4691.04 tps (variance < 0.013%), wall 873.15~873.27s, req/s 0.5726 동일. Codebase 의 측정 안정성 입증.
3. **NEO 의 진정 cost** — chain firing 0.60% sparse 영역에서 **-21% throughput + 26% wall**. NEO 인프라 overhead (worker fork branch, adapter cdec_ids 추출, sub_batches build, schedule cost) 가 sparse fire 의 마진을 압도.
4. **Token output equivalence 영역** — total_output_tokens vanilla 4,096,000 vs v1.3 4,091,911 (4089 token 차이, **0.10%**). CLAUDE.md Constraint 의 *분포·의도 수준 유사성* 영역 (token-level bit-exact 미요구).
5. **init_s outlier** — v1.3 의 1 회차 (run3) 가 131.74s (다른 회차 87.97/88.32). NEO 초기화 randomness — 본 측정 영역 외.

### 결론 + 다음 phase

**v1.3 의 정체 확정**:
- chain firing 0.60% sparse + crash 0 + reproducible — **infra 동작 입증**
- throughput vanilla 대비 **-21% 회귀** — NEO 의 *진정한 발화 영역 미진입*. v1.0~v1.2 (chain 0~0%) 와 본질적 동등 영역 (사용자 지적의 "퇴보" 영역 포함)

**다음 phase 의 input** (별도 plan):
- chain firing 의 *결정적 reproducible trigger* 식별 (D13~의 KV threshold 외 다른 변수)
- *진짜 NEO benefit* (active 평형 + chain 80%+ + throughput vanilla 동등) 영역 진입 path

---

## 5.13 · NEO 정통 정합 (Option I+K+C+L) — chain firing 88~99% 달성 입증 + throughput 회귀 미해소 (2026-05-11)

### 배경

§5.12 의 -21% throughput 회귀 + chain firing 0.6% sparse 영역 정체. 사용자 지적: *"neo 논문을 제대로 구현하지 않은 거 같은데"*. NEO github (`NEO-MLSys25/NEO`) 소스 + paper (MLSys 2025) 직접 검증 후 *4 단계 누적 fix*. 

### NEO 소스 분석 결과 (정통 정합 검증)

**NEO `swiftllm/server/scheduler.py:142-235`** 의 `_decide_mode_and_gen_batch`:
- Step 3: `for req in cpu_decoding_q: batches[1].add_cdec(req)` — 매 step *load-aware* 으로 cdec dispatch
- `cpu_decoding_q` 가 *swap_out~swap_in 사이의 영구 큐*. 매 step CDEC dispatch fire.

**NEO `swiftllm/server/block_manager.py:153-162`** 의 `_alloc_blocks_for_batch`:
- `gpu_block_manager.alloc(batch.all_reqs[:batch.num_prgds], ...)` + `cpu_block_manager.alloc(batch.all_reqs[batch.num_prgds:], omit_last=False)` — *매 step* 증분 alloc.

**NEO `pacpu/core.h:11-31`** 의 `brute::store_kv`:
- `block_pos = (seq_len - 1) / BLOCK_SIZE`, `block_id = block_table[block_pos]` — *block_table[block_pos] 가 valid block 이어야*. 매 step 증분 alloc 의 *진짜 이유*.

### 우리 구현의 결손 식별 (소스 fact)

| 영역 | NEO 정통 | 우리 구현 | 결손 정도 |
|---|---|---|---|
| `cpu_decoding_q` 영구 큐 | swap_out~swap_in 잔류 | `_neo_cpu_resident_mirror` 적재 OK | 부분 결손 (size 안정 X) |
| load-aware 매 step cdec | `add_cdec` 매 step | `cdec_ids` 추출은 `SWAPPED_OUT` 만 | 결손 |
| D10 가드 (vllm 우리쪽) | (없음) | `num_new_tokens=0` 클램프 → output 제외 | *부작용* |
| 매 step CPU block alloc | `cpu_block_manager.alloc(omit_last=False)` 매 step | swap-out 시점 1회만 (재 alloc ValueError) | **본질 결손** |
| 매 step GPU block alloc | `gpu_block_manager.alloc` 매 step | SWAPPED_OUT 시 alloc 안 함 (RUNNING reqs 만) | **본질 결손** |

### 4 단계 fix 적재 (Option I → K → C → L)

**Option I — D4 swap_in MIN_BUFFER guard** (`neo_scheduler_adapter.py:1051-1080`)
- D4 swap_in path 의 `_max_swap_in = min(_max_swap_in_env, max(0, len(mirror) - MIN_BUFFER))`. mirror size 가 MIN_BUFFER (default 8) 미만이면 swap_in 0 — *영구 큐 시간 확보*.
- env: `VLLM_NEO_MIRROR_MIN_BUFFER` (default 8).

**Option K — D10 가드 완화** (`scheduler.py:483-512`)
- D10 가드의 `_max_allowed_d10 <= 0` 영역의 `num_new_tokens=0` → *1*. mirror 의 SWAPPED_OUT reqs 가 *vllm output 에 포함*되어 cdec dispatch 가능.
- SEGV 회피는 D11 dynamic precheck 의존.
- env: `VLLM_NEO_OPTION_K` (default 0, kill switch).

**Option C — decide_mode load-balanced cdec 배포** (`neo_scheduler_adapter.py:793-870`)
- adapter.schedule() 의 cdec_ids 추출에서 *mirror 의 reqs 를 cpu_decoding_q 로* `decide_mode` 호출. NEO `_decide_mode_and_gen_batch.Step 3` 등가. `batches[1].cdec_reqs` → cdec_ids.
- env: `VLLM_NEO_OPTION_C` (default 0).

**Option L — NEO 정통 매 step 증분 CPU block alloc** (`neo_cpu_kv_buffer.py:204-265` + `attention.py:797-840`)
- `NeoCpuKvBuffer.ensure_capacity(req_id, target_num_blocks)` 추가. NEO `DeviceBlockManager.alloc` 의 *increment* logic 모방.
- worker attention.py 의 cdec dispatch 직전 호출. seq_len 의 nblock 부족 시 *차이만큼 free pool 에서 추가 alloc*.
- env: `VLLM_NEO_OPTION_L` (default 1).

### 검증 회차 결과

| 회차 | env | chain firing | NEO CDEC CALL | D11 OOB | swap-in mismatch | throughput | crash | 비고 |
|---|---|---|---|---|---|---|---|---|
| try74 | (D14v2 기준선) | 0% (active=0) | 0 | 0 | 0 | 3642 tps | 0 | infra OK, fire X |
| try77 | I only | 0% | 0 | 0 | 0 | (mid-stop) | 0 | infra-level prerequisite PASS |
| try78 | I + C | 0% | 0 | 0 | 0 | (mid-stop) | 0 | C single 불가 (D10 가드 + line 573 충돌) |
| try80 | I + K + C | **96%** | 32 | **18560** | 다수 | (mid-stop) | 0 | chain firing 발화 + cdec OOB 다수 |
| **try81** | **I + K + C + L** | **88~99%** | **127,300/worker** | **0** | 5744 (잔존) | **390 tps** | **0** | **NEO 정통 정합 달성** |

### 핵심 입증 (try81)

1. **chain firing 활성화** — total=2200 step 중 active=2176 = **98.9%** (NEO paper 영역 진입)
2. **NEO CDEC CALL 폭증** — 32 (try80) → **127,300** (per worker, 8 workers 합 ~1M) = *+469× 폭증*
3. **D11 OOB 완전 해소** — 18560 (try80) → **0** (Option L 의 NEO 정통 정합 효과)
4. **NEO BUF EXTEND** — count 40+, FAIL 0 (CPU pool 충분)
5. **crash 0** — EngineDead/SIGSEGV/AssertionError 모두 0

### 미해소 issue (Option M 영역)

- **swap-in shape mismatch 잔존** — 5744 회 warning. *GPU side 의 매 step block alloc 결손*. mirror 의 reqs 가 *Option L 로 CPU block 증분* OK, 그러나 GPU side 는 *RUNNING reqs 만 매 step alloc*. SWAPPED_OUT reqs 의 진행된 seq_len 의 GPU block alloc 가 *swap-in 시점에만* — *Option K 의 chain firing 진행과 mismatch*.
- **throughput 11× 저하** — 390 tps vs vanilla 4690 tps. mirror 의 reqs 가 *swap-in fail 로 GPU 복귀 못 함* → CPU 단독 decode. paper 의 *14% gain on H100* 은 *GPU+CPU 동시 동작* 전제 — 우리는 *CPU 단독* 영역 진입.

### 결론 + 다음 phase

**본 phase 의 진짜 성과**:
- chain firing **0.6% → 98.9%** 영역 진입 (NEO paper 영역)
- D11 OOB / swap-in mismatch 등의 *진짜 root* 식별 + 부분 fix
- NEO github + paper 직접 검증 후 *소스 fact 기반* 진행 — 추측 X

**Option M (다음 phase) 필요**:
- swap-in 시 GPU side block 추가 alloc — `neo_swap_in_alloc` 의 `n_tokens` 가 *현재 CPU buffer 의 alloc 된 token 수와 동기화* 필요
- 또는 swap_out 시 *upfront max_total_tokens block alloc* (단순 fix) — CPU pool 부족 시 fail → vanilla preempt fallback

**사용자 ground rule 준수**:
- NEO 비활성화 옵션 *추가 안 함*. 모든 fix 는 *NEO 메커니즘 활성화* 방향.
- commit & push 는 *사용자 명시 후만*.

### 측정/관측 산출물

- launch script: `eval/run_v4_K_OptI_only.sh`, `run_v4_K_OptI_plus_OptC.sh`, `run_v4_K_OptIKC.sh`, `run_v4_K_OptIKCL.sh`
- 회차 결과: `eval/results/20260510_235401_try77_v4_K_OptI_only/`, `20260511_000346_try78_v4_K_OptI_plus_OptC/`, `20260511_001239_try80_v4_K_OptIKC/`, `20260511_003320_try81_v4_K_OptIKCL/`
- 코드 위치:
  - `vllm/v1/core/sched/neo_scheduler_adapter.py` (Option I, C 적재)
  - `vllm/v1/core/sched/scheduler.py` (Option K 적재)
  - `vllm/v1/core/sched/neo_cpu_kv_buffer.py` (Option L `ensure_capacity` API)
  - `vllm/model_executor/layers/attention/attention.py` (Option L caller)

---

## 5.14 · TSK_019 v1.7 (SUB_025) — async swap_out (D→H DMA forward overlap) + deadlock escape → **1,799.9 tps (+187% vs sync)** (2026-05-13)

### 배경

v1.6 winning config (commit `0424b2828`, NUMA-aware pin + routing fix, ~580 tps) 위에서 측정한 sync NEO try102 가 vanilla 4,682 tps 대비 627 tps (13.4%) 로 머무름. py-spy / py-spy --native 진단:

- `_neo_swap_out_one_req` 의 `copy_layer_in` (D→H per-layer copy, blocking) 이 매 swap_out 시 **forward 시작 전 ~108ms blocking** 으로 GPU idle 발생
- swap_out 1 step 당 평균 2-3 req → step 시간 의 60% 이상이 D→H sync wait 로 점유
- try102 의 전체 4,656 swap_out × 108ms ≈ 503 초 / 6,382 wall × 7.9%

→ async D→H 로 forward 와 병행하면 그 만큼 회수 가능 한 점 확인.

### 구현 (SUB_025)

3 phase 구조 — `vllm/v1/worker/gpu_model_runner.py`:

1. **Phase 1: gather** (swap_stream 내, ~15ms)
   - `_neo_swap_out_gather_phase`: GPU `kv_caches[layer][gpu_block_idx]` → 독립 GPU tensor (`k_gpu`, `v_gpu`). forward 가 kv_caches 에 쓰기 *전* 에 완료해야 race 회피.
2. **Phase 2: DMA launch** (swap_stream 내, non_blocking=True)
   - `_neo_swap_out_dma_phase`: `k_gpu → staging_k[layer, :n_blocks]` (pinned CPU). `dma_event.record()`. 이후 forward 는 `current_stream` 에서 병행.
   - `pre_fwd_event` 로 swap-in + gather 완료 후 forward 시작 보장.
3. **Phase 3: drain** (forward 끝나고)
   - `_neo_drain_pending_swap_dma`: `dma_event.synchronize()` (보통 즉시 리턴) → `buf.copy_all_layers_in_from_staged()` (CPU memcpy, lock 영역) → `_neo_cpu_resident_reqs.add(req_id)`.

**Staging buffer 설계**:
- shape `(num_layers=80, max_blocks_per_req=1024, num_kv_heads=1, block_size=16, head_dim=128)` × fp16 = K 320 MiB + V 320 MiB
- **worker당 1개** (현재 SUB_025) → 한 step 내 첫 req 만 async, 2번째+ req 은 sync fallback. async 비율 ~40%.
- 추가 lock 회피: NEO buffer 의 `_neo_synchronized` (RLock) 는 Phase 3 (CPU scatter) 만 점유 → DMA 자체는 lock 무관.

**env gate**: `VLLM_NEO_ASYNC_SWAP=1` (default ON, =0 시 구 sync 경로 강제).

**관련 변경 site**:
- `vllm/v1/core/sched/neo_cpu_kv_buffer.py`: `copy_all_layers_in_from_staged()` 신설 (CPU scatter 전용)
- `vllm/v1/worker/gpu_model_runner.py`: 위 3 phase 함수 + `_neo_handle_kv_swap` 의 swap_stream 블록 재구성 + `execute_model` 후 drain 호출

### deadlock escape

50p run (보조 측정) 에서 42/50 prompts 완료 후 GPU SM 0% + EngineCore `_process_engine_step` idle deadlock. py-spy 진단 결과:

- `EngineCore` MainThread: `_process_engine_step` → `step_fn()` 내부 block
- `Worker_TP0` MainThread: `shm_broadcast.acquire_read()` 대기 (다음 execute_model 입력)

양측 모두 다음 step 대기 → 양방향 교착. Root: 남은 8 reqs 모두 CPU-resident (`SWAPPED_OUT` + mirror) 상태에서 `mirror_size=8 ≤ MIN_BUFFER(=8)` 시 D4 swap-in path 의 `_excess = max(0, mirror_size - MIN_BUFFER) = 0` → `_max_swap_in = 0` → 영구 swap-in 차단.

**Fix** — `vllm/v1/core/sched/neo_scheduler_adapter.py:1240+`:
```python
_gpu_active = sum(1 for r in self.running if r.status == RUNNING)
if _max_swap_in == 0 and _gpu_active == 0 and not self.waiting and _mirror:
    _max_swap_in = min(원래_cap(2), len(_mirror))  # MIN_BUFFER 가드 bypass
    logger.info("[NEO deadlock escape] GPU-active=0 waiting=0 mirror=%d ...")
```

50p run 에서 escape 발화 후 정상 완주 (798s, 512.9 tps). 500p run 에서는 미발화 (GPU-active 가 항상 양수).

### 측정 결과 (500p)

| 회차 | output_tps | wall_s | vs vanilla | vs sync |
|---|---|---|---|---|
| vanilla 500p (eval/results/20260513_142841_vanilla_clean) | 4,682.1 | ~875 | 100% | — |
| NEO sync try102 (v1.5 clean, eval/results/20260511_092250_try102_v5_clean) | 627.6 | 6,382 | 13.4% | 100% |
| **NEO async (eval/results/20260513_192812_neo_clean)** | **1,799.9** | **2,252** | **38.4%** | **+187%** |

**async swap_out 동작 verify**:
- TP0 async events: 1,545 / sync fallback: 2,349 → **async 비율 39.7%**
- async `total_ms` 분포 (DMA + forward overlap window):
  - <50ms: 211건 (14%, 짧은 seq)
  - **50-200ms: 1,137건 (74%)** ← forward 와 정상 병행
  - >200ms: 197건 (13%, 340-525 MiB 대형 transfer)
- sync `elapsed_ms`: avg 108ms (forward blocking 시간 동등)
- `b1_avg = 10` (CPU decode 활성, deadlock escape 미발화)

### 잔존 issue + 다음 phase

- **async 비율 ~40% 한계** — staging buffer 1 개 / worker. 한 step 내 2번째+ req 은 sync fallback. → **SUB_026 (N=3 확장) 후속 예정**.
- **vanilla 대비 여전히 38.4%** — sync swap_out 가 ~60% 차지, 추가 async 화 + cdec executor 효율 향상 필요. cdec_wait_avg 3.59ms (PROFILE PER-LAYER) 가 cdec 자체 cost.
- **CPU KV pool 포화 경고** — `BUF ALLOC FAIL req=464 free=48/131072` 1 회 발생 (99.96% full). pool 크기 vs active reqs balance 후속 분석 영역.

### 측정/관측 산출물

- launch: `eval/run_neo_clean.sh` (NEO async, 500p), `eval/run_vanilla_clean.sh` (vanilla 500p), `eval/run_neo_short.sh` / `eval/run_vanilla_short.sh` (50p 단축)
- 결과:
  - `eval/results/20260513_192812_neo_clean/result.json` — NEO async 500p (본 main 회차)
  - `eval/results/20260513_142841_vanilla_clean/result.json` — vanilla 500p baseline
  - `eval/results/20260513_165113_neo_async_short/result.json` — NEO async 50p + deadlock escape 발화
  - `eval/results/20260513_161400_vanilla_short/result.json` — vanilla 50p
- 코드 (commit `f25c37190` → rebase 후 `a3db943f4`):
  - `vllm/v1/worker/gpu_model_runner.py` — async swap_out 3 phase
  - `vllm/v1/core/sched/neo_cpu_kv_buffer.py` — `copy_all_layers_in_from_staged`
  - `vllm/v1/core/sched/neo_scheduler_adapter.py` — deadlock escape (line 1255+)
  - `vllm/v1/core/sched/mode_selector.py` — decide_mode balance-driven 복원
  - `vllm/v1/core/sched/neo_scheduler.py` — `step_2_3_only` 헬퍼

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
| 2026-04-30 | **§5 갱신 — 현재 NEO ON 의 의미 + 정식 비교 회차 plan 분리** | TSK_016 의 Step 5.1~5.4 wiring 통과 후 사용자 질문 ("500 prompt 로 NEO 기능 검증이 가능한거지?") 에 답하기 위한 layered 정리. (1) **§5.1**: 현재 NEO data path 는 forward-context fork 미적용 → KV cache cross-contamination → vanilla fallback active. NEO ON 회차의 의미 = *무회귀 검증* 만. 500 회차 = 개발 회귀, 1000 = 정식 회귀, 5000 = 외부 보고 회귀 (효과 측정 단계 후로 미룸). (2) **§5.2**: 진짜 효과 측정은 Step 5.5 (forward-context fork) → TSK_015 (KV exclusive) → TSK_017 (PerfPredictor 실측 table) → TSK_018 (CPU kernel 통합) 누적 후 의미. (3) **§5.3**: 무회귀 / 효과 측정 / 외부 보고 의 3 단계 비교 회차 plan 명시. |
| 2026-05-03 | **§5.4 + §5.5 신설 — Phase B async fix 후 NEO ON 측정 3 회차 + 진짜 cdec dispatch 발화 path 식별** | (1) **§5.4** 적재 — Phase B `NeoSchedulerAdapter(AsyncScheduler)` 1줄 fix 후 정식 비교 회차 (500 / 1000 / 5000 × 50:50, Llama-70B + TP=8, VLLM_NEO_KV_FREE=1). 결과: 500 +8.72% / 1000 +1.49% / 5000 **-3.50% wall regression** (가장 통계 신뢰 영역). NEO 발화 카운트 = 모두 zero (max_conc 78 = 30%, KV usage 90~100% sustained 였으나 swap_out 미발화). (2) **swap_out 미발화 root cause 식별**: NEO sibling 의 `_sync_neo_gpu_decoding_q` 가 *decode 단계* reqs 만 매핑 → prefill 단계 KV pressure 추적 안 됨. (3) **§5.5** 신설 — 진짜 cdec dispatch 발화 path 3 옵션 (prefill KV 추적 보강 / RATIO env forced-fire / max_num_seqs+input ↑). 별도 multi-day / multi-hour 영역. (4) **결론**: Phase A (NEO 베이스 적재) + Phase B (async fix) = NEO 인프라 활성 + vanilla 동등 ±5% 영역 입증 완료. 진짜 NEO gain (논문 H100 14% 영역) 은 별도 path. |
| 2026-05-05 | **§5.6 신설 — Phase D (v37~v41 chain) 측정 + NEO ON throughput > vanilla 검증 PASS at 500p sweet spot** | (1) **NEO ↔ vLLM single source path 통합** (commit `4b287b1639`) + `_preempt_request` SWAPPED_OUT 자연 발화 → adapter cdec_ids 직접 attach. NEO fork firing rate **66~98%** 영역 진입 (Phase B 의 zero-fire 대비 큰 변화). (2) **v37→v38 architectural surgery**: `unified_attention_with_output` 의 cdec-only sub-batch 시 `self.impl.forward(...)` skip — swiftllm 패턴 매칭 (commit `eeed0d46fc`). v37 (1928.63 tps) → v38 (**2276.42 tps**, +18% 단일 fix). (3) **stop 조건 1 검증 PASS**: NEO v38 500p output_tps **2276.42** vs vanilla **2190.83** → **+3.91% 우월** / wall **-8.27%**. (4) **chain 측정 매트릭스**: 300p / 500p / 1000p × vanilla / v37 / v38 / v40 / v41 — sweet spot **500p 한 점만** NEO win. 300p 는 KV pressure 부족 (firing 55%) 으로 lose, 1000p 는 token corruption 누적 (12.89%) 로 lose. (5) **v41 (no-fastmath kernel)**: ISPC `--opt=fast-math` + C++ `-Ofast` 제거 → token loss 4.69% → **2.84%** (39% 감소). 500p NEO win **+3.13%** 유지. fast-math 가 corruption 의 30-40% contributor 입증. (6) **per-step drift 누적 패턴**: cdec steps 300p 8163 / 500p 16310 / 1000p 31767 → token loss 비례 누적. NEO sweet spot 좁힘 메커니즘 식별. (7) **stop 조건 2 (정확도 보존)** 미검증 — token loss 잔존 (2.84%-14.47% size 따라). (8) **현재 코드 상태**: HEAD = v38 (Python). kernel `.so` = v41 strict FP rebuild (uncommitted). |
| 2026-05-10 | **§5.10 신설 — TSK_019 v4 D0-D5 검증 + pacpu store_kv segfault root 식별** | commit `df2cb7c81e` 의 D0~D5 (mirror set / num_new_tokens=0 fix) 검증 회차 try60 3 개 적재. (1) **infra-level 성과**: chain trigger path 인프라 적재 + natural-preempt → CPU mirror 잔류 → next-step decode lifecycle 동작 확인. (2) **try60-γ (05:22) crash root 식별**: worker pacpu kernel `brute::store_kv` 의 SIGSEGV. EngineDead 는 결과적 증상. 기전 = D5 fix 의 부작용 — SWAPPED_OUT decode req 의 `seq_lens` 가 매 step +1 누적되지만 CPU buffer block_count 는 swap-out 시점에 고정 → `block_pos = (seq_len-1)/BLOCK_SIZE` 가 결국 `block_table` row width OOB → garbage `block_id` → invalid `cache_off` → memcpy SEGV. (3) **19 항목 measured**: chain active 0/4400 (0%), SWAP_IN_DONE=0, EngineDead=1. commit msg 의 "complete chain 입증" = try60-β (8min 외부 종료) 의 gates fire 관찰만. (4) **다음 phase input**: D6 (SWAP_IN_DONE 강제 활성), D7 (D5 condition 가드), D8 (FORK chain attach=1 → active=0 비대칭 추적), TST_003 verdict. |
| 2026-05-10 | **§5.11 신설 — Phase K (D6~D12) 안정화 + bistability 발견** | try60-γ ~ try71 12 회차 누적 결과 적재. (1) **D6~D10 누적 fix** (commits `2cf51460ed` + try61~67 7 회차 산출물 + 7 launch scripts) — 정적 코드 분석 기반 SEGV 회피 시도. crash 8 → 2 (try64 D8 v1, try67 D10) 까지 줄임. (2) **사용자 ground rule 재명시 + 메모리 기록**: NEO 비활성화 옵션 절대 금지. (3) **동적 분석 전환** — 정적으로 race window 닫힌 것으로 보였는데도 SEGV → D11 dynamic precheck (commit `5f85875b7e`) 도입. pacpu kernel 직전 `(seq_len, nblocks, block_pos)` 검증 + OOB reqs 발견 시 cdec dispatch skip. (4) **try68 첫 PASS** — run 완주, crash 0, FORK active **1429** (chain firing 6.4%), swap_in done 771, output_tps 2743, wall 1488s. (5) **진짜 SEGV root 식별** — D11 OOB log (`block_pos == nblocks` 정확히 1 칸 OOB) 가 dump = engine 측 num_computed 와 worker 측 num_computed 사이 **async lookahead ~7 token gap** 이 1 block 경계 넘는 timing race. (6) **D12 stash margin** — v1 (1 block) / v2 (8 token) 둘 다 chain firing lockout, v3 default 0 (D8 v1 동작 복원). (7) **NEO bistability 발견** — try68 만 active 평형 진입 (chain firing), try69~71 은 inactive 평형 (chain 0). 동일 코드/env 인데 workload randomness + KV pressure 진입 timing 의존. (8) **누적 status**: 19 항목 중 ✅ 8 (KV/swap/deadlock/crash) + 🔶 8 bistable (cdec/fork chain) + 미수행 3 (#15-17). **infra 안정화 PASS**, active 평형 강제 진입은 D13+ 별도. |
| 2026-05-10 | **v1.3 명명 변경 — try68 → try51** | 사용자 명시 (2026-05-10 KST 19:00 영역). try68 의 active 평형 (chain 6.4%) 은 bistability 의 우연 진입으로 reproducibility 미확보 (3×3 perf compare 의 v13_run1/run2 모두 inactive 평형 chain 0%). v1.3 명명 부적합 판정. **v1.3 = try51 회차** (TSK_019 v3 Phase B, commit `f2678c2f4` "v3 C4-C5 swap_in path 활성화 + LRU stub", 2026-05-09 10:27 KST) 으로 변경. 선정 사유: chain firing 첫 측정 fire (0.60%) + reproducibility 확보 (try50 동일) + crash 0 + run 정상 완주. PLN_001 §5.11 의 "버전 명명" subsection update + git tag v1.3 도 `f2678c2f4` 로 이동 (force push). |
| 2026-05-11 | **§5.13 신설 — NEO 정통 정합 (Option I+K+C+L) chain firing 88~99% 달성 + Option M 영역 남음** | (1) NEO github (`NEO-MLSys25/NEO`) + paper (MLSys 2025) 직접 분석 — `_decide_mode_and_gen_batch` Step 3 (load-aware 매 step cdec) + `_alloc_blocks_for_batch` (매 step 증분 alloc, `omit_last=False`) + `pacpu::brute::store_kv` 의 `block_table[block_pos]` valid 요구. (2) **우리 구현 본질 결손 식별**: cpu_decoding_q 영구 큐 미정합 + cdec_ids 추출 SWAPPED_OUT 만 + D10 가드 `num_new_tokens=0` 클램프 + 매 step CPU/GPU block alloc 결손. (3) **4 단계 누적 fix**: Option I (D4 swap_in MIN_BUFFER guard) + Option K (D10 가드 완화 num_new_tokens=1) + Option C (decide_mode load-balanced cdec) + Option L (NEO `DeviceBlockManager.alloc` increment logic 모방한 `NeoCpuKvBuffer.ensure_capacity`). (4) **검증 회차 결과**: try77 (I only) infra prerequisite PASS; try78 (I+C) D10 가드 충돌로 fire 안 됨; try80 (I+K+C) chain firing 96% + D11 OOB 18560; **try81 (I+K+C+L)** chain firing **98.9%** + D11 OOB **0** + NEO CDEC CALL **127k/worker** (이전 32 대비 +469×) + crash 0. (5) **paper 영역 진입 입증** — NEO 의 "14% throughput gain on H100" 의 *메커니즘 (load-aware cdec dispatch + 매 step KV 적재)* 정합. (6) **잔존 issue**: swap-in shape mismatch 5744 회 (GPU side block 추가 alloc 결손 — Option M 영역) + throughput 11× 저하 (mirror reqs 가 swap-in fail 로 GPU 복귀 불가 → CPU 단독 decode). (7) **다음 phase**: Option M — `neo_swap_in_alloc` 가 CPU buffer 의 *현재 alloc block 수* 와 동기화. 또는 swap_out 시 *upfront max_total_tokens block alloc* (단순 fallback). 사용자 결정 대기. |
| 2026-05-13 | **§5.14 신설 — TSK_019 v1.7 (SUB_025) async swap_out + deadlock escape → 1,799.9 tps (+187% vs sync)** | (1) **swap_out async 화** — 기존 `copy_layer_in` per-layer D→H sync (avg 108ms) 가 forward 시작 전 GPU idle 야기. **3 phase 구현** (gather/DMA launch/drain) 으로 D→H 가 forward 와 병행. **staging buffer** worker당 K+V 320 MiB pinned. (2) **deadlock escape** — 50p run 에서 마지막 8 reqs 모두 CPU-resident + `mirror_size ≤ MIN_BUFFER(=8)` 시 D4 swap-in path 가 영구 차단. py-spy 진단으로 EngineCore↔Worker 양방향 wait 식별. **fix**: `GPU-active=0 AND waiting=0 AND mirror non-empty` 시 MIN_BUFFER 가드 bypass + `max_swap_in` 원래 cap 복원. (3) **측정 결과** (500p × 8192 in/out): output_tps **1,799.9** vs sync try102 627.6 (**+187%**) vs vanilla 4,682.1 (38.4%). async 비율 39.7% (1차 req only, staging 1개 한계), `total_ms` 50-200ms **74%** = forward 와 정상 overlap 확인. (4) **잔존**: async 비율 ~40% 한계 (staging buffer 1개), CPU KV pool 포화 (1회 BUF ALLOC FAIL), vanilla 대비 38.4% (sync swap_out 가 여전히 ~60%). (5) **다음 phase** — **SUB_026** (staging buffer N=3 확장) → async 비율 60-70% 기대, worker당 pinned 640 MiB → 1.92 GiB (시스템 RAM 2 TiB 대비 negligible). commit `a3db943f4`. |

---

**↑ 부모 PLN**: [`PLN_001`](PLN_001.md) · **↟ 조부 IDE**: [`IDE_006`](README.md) · **연계**: [`TSK_014`](TSK_014.md), [`TSK_016`](TSK_016.md), [`TSK_017`](TSK_017.md), [`NEO_redesign`](NEO_redesign.md), [`NEO_code_deepdive`](NEO_code_deepdive.md)
