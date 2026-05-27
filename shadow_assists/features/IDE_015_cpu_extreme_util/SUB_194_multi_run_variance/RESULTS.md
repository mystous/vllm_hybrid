# SUB_194 — Top-3 net positive lever multi-run variance verification

> **parent**: paper §4 binding validation (SUB_183/188/190 1-run signal 검증).
> **scope**: 2026-05-27 14:33 ~ 15:05 KST (~32 min wall, 3 lever × 2 mode × 3 run = 18 runs).
> **status**: ✅ 완료 — **충격적 finding**: 1-run signal 모두 cold-start variance 안에 묻힘. SUB_190 tokenize 는 multi-run mean 으로 **부호 반전 (−5.96%)**.

---

## 0. 두괄식 — 1-run signal **신뢰성 의문** ⚠⚠

| Lever (parent SUB) | 1-run Δ (SUB_173~190) | **3-run mean Δ** | 3-mix avg stddev (propagated) | warm-only Δ (run1 제외) |
|---|---:|---:|---:|---:|
| **L183 NUMA pin** | +1.54% | **+2.24%** | ±35.08 pp | **+3.13%** |
| **L188 softmax precompute** | +1.84% | **+0.53%** | ±34.67 pp | **+0.81%** |
| **L190 async tokenizer** | +1.66% | **−5.96%** ⚠ | ±31.85 pp | **−6.40%** ⚠⚠ |

**run1 cold-start outlier 영향**: run1 의 3-mix avg AGSD = ~2,800 tps (cudagraph compile 진행 중), run2/run3 = ~4,300-4,600 tps (cache warm). variance 가 magnitude 보다 훨씬 커서 1-run signal 의 신뢰성 무너짐.

**warm-only mean (run2/run3)** 으로 보면:
- L183 NUMA: +3.13% (1-run +1.54% 와 같은 방향, magnitude 2× 증가)
- L188 softmax: +0.81% (noise floor)
- L190 tokenize: **−6.40%** (1-run +1.66% 와 부호 반전, regression!)

---

## 1. 측정 setup

| 항목 | 값 |
|---|---|
| canonical | Qwen 32B TP=4×2, max-tokens=32, conc=32, 500 prompt × 3 mix |
| scenario | agsd-gated only (vanilla/trident skip — agent 시간 절약) |
| levers | L183 NUMA pin / L188 softmax precompute / L190 tokenize worker |
| runs per (lever × mode) | 3 |
| total cycles | 3 lever × 2 mode (off/on) × 1 vllm boot = 6 boots |
| 각 cycle | boot 80s + 3 × (3 mix × ~30s) ≈ 350s (~6 min) |

각 vllm boot 후 3-run back-to-back 측정 — run1 의 cudagraph cold-start 영향이 isolated.

## 2. 상세 결과

### 2.1 L183 NUMA pin (3-mix avg AGSD)

| run | OFF tps | ON tps |
|---:|---:|---:|
| 1 | 2,828.7 | 2,814.5 |
| 2 | 4,330.7 | 4,467.4 |
| 3 | 4,497.2 | 4,636.0 |
| **mean ± stddev** | **3,885.5 ± 919.0** | **3,972.6 ± 1,006.5** |
| Δ% | — | **+2.24%** (propagated ±35.08 pp) |
| warm-only (run2/3) | 4,413.5 | 4,551.7 | **+3.13%** |

→ run2/3 만 보면 +3.13% — **3 lever 중 가장 robust 한 net positive**. 1-run signal (+1.54%) 의 방향성 일치, magnitude 2× 증가.

### 2.2 L188 softmax precompute

| run | OFF tps | ON tps |
|---:|---:|---:|
| 1 | 2,809.9 | 2,799.9 |
| 2 | 4,316.2 | 4,402.3 |
| 3 | 4,545.6 | 4,531.9 |
| **mean ± stddev** | **3,890.5 ± 942.9** | **3,911.4 ± 964.8** |
| Δ% | — | **+0.53%** (propagated ±34.67 pp) |
| warm-only | 4,430.9 | 4,467.1 | **+0.81%** |

→ 1-run signal (+1.84%) 의 30% 만 유지. multi-run mean 으로 보면 **noise floor**.

### 2.3 L190 async tokenizer

| run | OFF tps | ON tps |
|---:|---:|---:|
| 1 | 2,790.8 | 2,663.5 |
| 2 | 4,307.6 | 4,027.3 |
| 3 | 4,409.3 | 4,131.1 |
| **mean ± stddev** | **3,835.9 ± 906.5** | **3,607.3 ± 819.0** |
| Δ% | — | **−5.96%** ⚠ (propagated ±31.85 pp) |
| warm-only | 4,358.5 | 4,079.2 | **−6.40%** ⚠⚠ |

→ 1-run signal (+1.66%) 과 **완전 부호 반전**. 본 lever 는 multi-run 으로 보면 net negative — **regression**.

## 3. cold-start variance 의 root cause

run1 의 ~2,800 tps vs run2/3 의 ~4,300-4,600 tps = **36-40% 감소**:
- vllm boot 직후 cudagraph PIECEWISE compile 이 진행 중
- piecewise compile range 의 inner shapes 가 첫 batch 들에서 trigger
- compile time 이 throughput 측정 wall 의 dominant fraction
- 80s boot 후 추가로 ~2-3 분 cudagraph compile + warmup 필요

이를 measurement 의 binding 지표로 가져가려면:
- **run1 외 (warm-only) mean 사용**
- **또는 첫 90s 의 timeshift 후 측정 시작**

## 4. paper §4 narrative 갱신 (큰 폭 update)

이전 narrative (1-run signal):
- lever 17 시도 중 paper-bound net positive 3 (SUB_183 +1.54% / SUB_188 +1.84% / SUB_190 +1.66%)
- paper §4 secondary lever 영역 small-positive 3개

**SUB_194 후 narrative**:
- L183 NUMA pin: **multi-run mean +2.24% / warm-only +3.13%** ⭐ — 가장 robust
- L188 softmax: multi-run +0.53% / warm-only +0.81% — noise floor
- L190 tokenize: multi-run **−5.96%** / warm-only −6.40% — **regression**, 1-run signal retract 권고
- 신뢰성 가능 net positive **1개만 (L183 NUMA pin)**, paper main 기준 +5% 여전히 미달

→ paper §4 의 small positive narrative 도 SUB_190 부분 retract. binding 지표는 multi-run mean.

## 5. 누적 패턴 update

| 카테고리 | 시도 | net positive (multi-run binding) | net loss | noise |
|---|---:|---:|---:|---:|
| 전체 lever | 18 | **1** (L183 +2.24-3.13%) | 6 | 11 |
| paper main 기준 (+5%) | 18 | **0** | — | — |

paper main lever 후보 = SUB_187 AMX draft real spec_decode integration 만 남음 (SUB_198 측정 예정).

## 6. 후속 SUB

- **SUB_190 tokenize lever 의 1-run signal retract note** RESULTS.md 에 추가 권고
- SUB_188 / SUB_173 / SUB_175 / SUB_183 등 1-run small positive 의 multi-run 검증도 paper-bound 가치 (별도 SUB)
- **measurement 의 default rule 변경 권고**: 1-run rule 의 한계 명시 — small magnitude signal (|Δ|<3%) 은 multi-run mean (3-run minimum) 또는 warm-only (run2+) 가 binding

## 7. raw data

- `measurements/{L183,L188,L190}/{off,on}/run{1,2,3}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (54 cells)
- `_monitor_L{183,188,190}_{off,on}_{cpu,gpu}.csv` (각 cycle 별 monitor)
- `logs/main_L{183,188,190}_{off,on}.log`
- `launcher.sh` + `runner.sh` + `aggregate.py` + `bench_agsd_only.py`
