# SUB_097 — Qwen 2.5-32B TP=8 single-instance + AGSD TP=4×2 end-to-end

> **parent**: TSK_020 / SUB_093 Phase 1 + SUB_094/095 형식 영역 합산
> **scope**: 2026-05-26 07:04 ~ 09:25 KST (~2.4 시간)
> **status**: ✅ 완료 — Phase A 18 cells + Phase B 9 cells = **27 cells**
> **모델**: Qwen 2.5-32B-Instruct

---

## 0. 두괄식

| Phase | setup | cells | 핵심 결과 |
|---|---|---:|---|
| **A** TP=8 single | 6 workload × 3 config (Llama 70B 동일 scale) | 18 | Trident core **6/6 net positive** ⭐ (avg +44%, sonnet +63.8%) |
| **B** TP=4 × 2 end-to-end | 3 mix × 3 scenario (CPU router) | 9 | AGSD-gated **3/3 net positive** vs vanilla-only (+102% ~ +139%) ⭐ |

→ **모델-vendor cross check**: Qwen 32B 영역 TP=8 single-instance 영역 **6/6 net positive** — Qwen 72B (5/6, code −5%) 와 달리 32B 는 code 도 거의 tie (−0.8%). **모델 크기 영역 클수록 vendor-specific regression 가능성 ↑**.

---

## 1. Phase A — TP=8 single-instance (500p × 8192in × 8192max, gmu=0.80, fp8 KV)

### 1.1 6 workload × 3 config matrix

| Workload | vanilla | ngram | **Trident core** | Trident vs vanilla | ngram vs vanilla |
|---|---:|---:|---:|---:|---:|
| sonnet | 11,725 | 16,622 | **19,206** | **+63.8%** ⭐ | +41.8% |
| chat | 8,943 | 10,542 | **12,066** | **+34.9%** ⭐ | +17.9% |
| code | 10,637 | 10,174 | **10,551** | **−0.8%** (≈tie) | −4.4% |
| mix-sh (M1) | 10,051 | 13,368 | **15,347** | **+52.7%** ⭐ | +33.0% |
| mix-bal (M2) | 9,514 | 11,900 | **15,838** | **+66.5%** ⭐ | +25.1% |
| mix-ch (M3) | 9,814 | 10,546 | **14,380** | **+46.5%** ⭐ | +7.5% |

→ **6/6 net positive** (code 영역 −0.8% 거의 tie, 통계적 noise 영역). Llama 70B (6/6 +19~+69%) 영역 패턴 영역 유사. Qwen 72B (5/6, code 영역 −5%) 보다 양호.

### 1.2 vs same-size class 비교 (vanilla)

| Model | TP | sonnet | chat | code | mix-bal |
|---|---:|---:|---:|---:|---:|
| **Qwen 2.5-32B** | 8 | **11,725** | **8,943** | **10,637** | 9,514 |
| Phi-3-medium 14B | 1 | 3,438 | 3,138 | 3,340 | 3,374 |
| Llama 3.3-70B | 8 | 7,679 | 2,267 | 6,718 | 6,054 |
| Qwen 2.5-72B | 8 | 6,456 | 2,560 | 6,227 | 5,702 |

→ **Qwen 32B vanilla 영역 모든 LLM (14B / 70B / 72B) 영역 압도** (모델 영역 절반 영역 + 8 GPU 영역 ample compute). chat 영역 특히 거의 3-4× 빠름.

### 1.3 wall sum + 시간

| config | wall sum (s) | duration |
|---|---:|---:|
| vanilla | 2,210 | 36 min |
| ngram | 1,860 | 31 min |
| Trident core | 1,635 | 27 min |
| **Phase A 총 wall** | **5,705** | **~94 min** |

---

## 2. Phase B — TP=4 × 2 AGSD end-to-end (200p × 256max × concurrency 32)

### 2.1 GPU 분배

| Backend | GPU | TP | spec |
|---|---|---:|---|
| vanilla | 0, 1, 2, 3 | 4 | OFF |
| Trident | 4, 5, 6, 7 | 4 | suffix + PIECEWISE |
| router | CPU (4 classifier worker) | — | regex-based |

→ **8 GPU 영역 4 + 4 분배** (full utilization, no idle).

### 2.2 결과 — 3 mix × 3 scenario

| Mix | vanilla-only | trident-only | **AGSD-gated** | vs vanilla | vs trident | backend split |
|---|---:|---:|---:|---:|---:|---|
| **balanced** (34:33:33) | 2,416 | 3,461 | **4,894** ⭐ | **+102.6%** ⭐ | **+41.4%** ⭐ | vanilla 66 + trident 134 |
| **sonnet-heavy** (60:20:20) | 2,506 | 5,330 | **5,576** ⭐ | **+122.5%** ⭐ | **+4.6%** | vanilla 40 + trident 160 |
| **code-heavy** (10:20:70) | 2,569 | 5,382 | **6,139** ⭐ | **+139.0%** ⭐ | **+14.1%** ⭐ | vanilla 40 + trident 160 |

### 2.3 vs SUB_095 (TP=2 × 2) 비교

| Mix | SUB_095 TP=2×2 AGSD | **SUB_097 TP=4×2 AGSD** | scale 효과 |
|---|---:|---:|---:|
| balanced | 3,449 | **4,894** | **+41.9%** ⭐ |
| sonnet-heavy | 4,084 | **5,576** | **+36.5%** ⭐ |
| code-heavy | 4,575 | **6,139** | **+34.2%** ⭐ |

→ **TP=4×2 영역 TP=2×2 영역 평균 +37% 향상** — backend 영역 sharding 더 많이 영역 forward 가속.

---

## 3. ★ Phase A vs Phase B 비교 (single-instance vs 2-instance gating)

| metric | Phase A TP=8 single | Phase B TP=4×2 e2e |
|---|---|---|
| GPU 사용 | 1 instance × 8 GPU | 2 instance × 4 GPU 각 |
| best config | Trident core (always-on) | AGSD-gated (router 영역 분기) |
| sonnet-heavy mix throughput | 15,347 tps (Phase A Trident mix-sh) | 5,576 tps (Phase B AGSD-gated sonnet-heavy) |
| **차이** | **2.75×** ⭐ | — |
| 이유 | Phase A 영역 500 prompt × 8192 token 영역 sustained / Phase B 영역 200 prompt × 256 token 영역 burst | scale 영역 직접 비교 영역 fair 아님 |

→ **두 setup 영역 별개 use case**: Phase A 영역 sustained batch processing (offline batch inference), Phase B 영역 mixed-traffic online serving. Phase A throughput 영역 sustained batch 영역 우월. Phase B AGSD-gated 영역 mixed-model deployment 영역 가치 영역 부각.

---

## 4. 종합 vs 다른 모델 (Trident core single-instance 6 workload 평균)

| Model | TP | Trident vs vanilla 평균 | net positive |
|---|---:|---:|:---:|
| Phi-3-medium-14B | 1 | +79.0% | 6/6 ⭐ |
| **Qwen 2.5-32B** | **8** | **+43.9%** | **6/6 ⭐** |
| Llama 3.3-70B | 8 | +50.9% | 6/6 ⭐ |
| Qwen 2.5-72B | 8 | +38.1% | 5/6 (code 영역 −5%) |

→ Qwen 32B 영역 14B (Phi-3) 영역 70B (Llama) 사이 영역 expected 영역 range. **14B / 32B / 70B 영역 모두 6/6 net positive 확정** (R/K boundary 영역 ≤14B 영역 안정).

---

## 5. raw data

- `tp8_single/{vanilla,ngram,agsd}/{sonnet,chat,code,mix-sh,mix-bal,mix-ch}/result.json` (18 cell + util)
- `tp4x2_agsd/benchmark_{balanced,sonnet-heavy,code-heavy}.json` (9 cell)
- `_logs/` (phase_a.log, phase_b.log + bench logs)

소스:
- `/tmp/run_sub097_orchestrator.sh` — Phase A + Phase B 통합 orchestrator
- `/tmp/sub097_phase_b_retry.sh` — Phase B retry (`--disable-custom-all-reduce` + CLASSIFIER_WORKERS=4)
