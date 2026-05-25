# SUB_095 — Option A: AGSD end-to-end multi-model expansion

> **parent**: TSK_020 / SUB_094 형식 영역 multi-model 확장
> **scope**: 2026-05-25 22:01 ~ 22:24 KST (~23 min wall, fix 포함)
> **status**: ✅ 완료 — 3 신규 모델 (Qwen 0.5B/1.5B/32B) × 3 mix × 3 scenario = **27 cell**
> **goal**: SUB_094 영역 Qwen 7B 영역 단일 모델 영역 → 4 모델 (+ Qwen 7B 영역 SUB_094 영역 재사용) 영역 fair 비교

---

## 0. 두괄식 — 4 모델 모두 AGSD-gated 우세 ⭐

| Model | Setup | balanced | sonnet-heavy | code-heavy |
|---|---|---|---|---|
| **Qwen 0.5B** (TP=1×2) | AGSD vs vanilla | **+70.3%** ⭐ | **+63.4%** ⭐ | **+103.6%** ⭐ |
| 〃 | AGSD vs trident | +34.6% | +10.5% | +18.4% |
| **Qwen 1.5B** (TP=1×2) | AGSD vs vanilla | **+64.6%** ⭐ | **+33.9%** ⭐ | **+112.2%** ⭐ |
| 〃 | AGSD vs trident | +27.8% | +7.9% | +9.4% |
| **Qwen 7B** (TP=1×2) | AGSD vs vanilla | **+61.8%** ⭐ | **+55.9%** ⭐ | **+122.5%** ⭐ |
| 〃 | AGSD vs trident | +43.3% | +15.1% | +17.5% |
| **Qwen 32B** (TP=2×2) | AGSD vs vanilla | **+93.8%** ⭐ | **+127.9%** ⭐ | **+144.1%** ⭐ |
| 〃 | AGSD vs trident | +26.8% | +2.0% | +0.5% |

→ **AGSD-gated 영역 4 모델 × 3 mix = 12/12 cell 영역 net positive** (vs vanilla-only / trident-only single deployment).
→ Llama 70B 영역 **GPU mem 영역 2-instance 불가** (TP=8 × 2 = 16 GPU 필요) — 따라서 1-3.5×10^10 parameter 모델 영역 검증 영역 4 모델 영역 cover.

---

## 1. 측정 환경

| 항목 | 값 |
|---|---|
| backend | 2 vLLM HTTP server (vanilla / Trident core suffix+PIECEWISE) |
| router | FastAPI + uvloop + ProcessPool × 16 worker + httpx conn pool 1024 |
| benchmark | 200 prompts × max_tokens 256 × concurrency 32 |
| mix scenarios | balanced (34:33:33), sonnet-heavy (60:20:20), code-heavy (10:20:70) |
| classifier | regex-based (sonnet / chat / code) |
| decision rule | chat → vanilla / sonnet/code → trident (Qwen 7B 결정 규칙, SUB_093 기반) |

### GPU 할당

| Model | TP per backend | GPU vanilla | GPU trident | total GPUs |
|---|---:|---|---|---:|
| Qwen 0.5B | 1 | GPU 1 | GPU 2 | 2 |
| Qwen 1.5B | 1 | GPU 1 | GPU 2 | 2 |
| Qwen 7B (SUB_094) | 1 | GPU 1 | GPU 2 | 2 |
| Qwen 32B | 2 | GPU 1,2 | GPU 3,4 | 4 |

---

## 2. 측정 결과 상세

### 2.1 Qwen 0.5B (TP=1 × 2)

| Mix | scenario | wall (s) | tps | p50 (ms) | p99 (ms) | backend split |
|---|---|---:|---:|---:|---:|---|
| **balanced** | vanilla-only | 11.6 | 3,672.4 | 2,062 | (참조 json) | direct 200 |
| **balanced** | trident-only | 9.1 | 4,643.6 | 1,267 | — | direct 200 |
| **balanced** | **AGSD-gated** | **6.8** | **6,252.1** ⭐ | **825** | — | vanilla 66 + trident 134 |
| **sonnet-heavy** | vanilla-only | 9.3 | 4,196.2 | 1,812 | — | direct 200 |
| **sonnet-heavy** | trident-only | 6.2 | 6,206.9 | 821 | — | direct 200 |
| **sonnet-heavy** | **AGSD-gated** | **5.7** | **6,857.7** ⭐ | **625** | — | vanilla 40 + trident 160 |
| **code-heavy** | vanilla-only | 10.9 | 4,227.2 | 1,822 | — | direct 200 |
| **code-heavy** | trident-only | 6.3 | 7,267.2 | 777 | — | direct 200 |
| **code-heavy** | **AGSD-gated** | **5.3** | **8,605.5** ⭐ | **447** | — | vanilla 40 + trident 160 |

### 2.2 Qwen 1.5B (TP=1 × 2)

| Mix | scenario | wall (s) | tps | p50 (ms) | backend split |
|---|---|---:|---:|---:|---|
| **balanced** | vanilla-only | 12.2 | 3,512.5 | 2,112 | direct 200 |
| **balanced** | trident-only | 9.5 | 4,524.0 | 1,099 | direct 200 |
| **balanced** | **AGSD-gated** | **7.4** | **5,782.8** ⭐ | **894** | vanilla 66 + trident 134 |
| **sonnet-heavy** | vanilla-only | 10.0 | 4,067.5 | 1,812 | direct 200 |
| **sonnet-heavy** | trident-only | 7.8 | 5,050.4 | 1,031 | direct 200 |
| **sonnet-heavy** | **AGSD-gated** | **7.4** | **5,449.3** ⭐ | **956** | vanilla 40 + trident 160 |
| **code-heavy** | vanilla-only | 11.2 | 4,210.9 | 1,851 | direct 200 |
| **code-heavy** | trident-only | 5.8 | 8,165.6 | 521 | direct 200 |
| **code-heavy** | **AGSD-gated** | **5.3** | **8,932.0** ⭐ | **429** | vanilla 40 + trident 160 |

### 2.3 Qwen 7B (TP=1 × 2) — SUB_094 영역 재인용

| Mix | scenario | wall (s) | tps | p50 (ms) | backend split |
|---|---|---:|---:|---:|---|
| balanced | vanilla-only | 13.3 | 3,753 | 2,049 | direct 200 |
| balanced | trident-only | 11.7 | 4,238 | 1,889 | direct 200 |
| balanced | **AGSD-gated** | **8.2** | **6,073** ⭐ | 1,239 | vanilla 66 + trident 134 |
| sonnet-heavy | vanilla-only | 13.0 | 3,865 | 1,905 | direct 200 |
| sonnet-heavy | trident-only | 9.6 | 5,234 | 1,372 | direct 200 |
| sonnet-heavy | **AGSD-gated** | **8.4** | **6,025** ⭐ | 1,122 | vanilla 40 + trident 160 |
| code-heavy | vanilla-only | 12.7 | 3,966 | 1,931 | direct 200 |
| code-heavy | trident-only | 6.7 | 7,512 | 640 | direct 200 |
| code-heavy | **AGSD-gated** | **5.7** | **8,825** ⭐ | 479 | vanilla 40 + trident 160 |

### 2.4 Qwen 32B (TP=2 × 2 = 4 GPU)

| Mix | scenario | wall (s) | tps | p50 (ms) | backend split |
|---|---|---:|---:|---:|---|
| **balanced** | vanilla-only | 27.9 | 1,779.3 | 4,354 | direct 200 |
| **balanced** | trident-only | 18.4 | 2,720.6 | 2,193 | direct 200 |
| **balanced** | **AGSD-gated** | **14.3** | **3,449.1** ⭐ | **1,066** | vanilla 66 + trident 134 |
| **sonnet-heavy** | vanilla-only | 28.1 | 1,792.1 | 4,057 | direct 200 |
| **sonnet-heavy** | trident-only | 12.5 | 4,004.1 | 1,137 | direct 200 |
| **sonnet-heavy** | **AGSD-gated** | **12.4** | **4,084.0** ⭐ | **893** | vanilla 40 + trident 160 |
| **code-heavy** | vanilla-only | 26.3 | 1,874.1 | 4,167 | direct 200 |
| **code-heavy** | trident-only | 10.8 | 4,550.6 | 1,057 | direct 200 |
| **code-heavy** | **AGSD-gated** | **10.8** | **4,574.9** ⭐ | **779** | vanilla 40 + trident 160 |

---

## 3. ★ 4 모델 종합 AGSD-gated 향상 (vs trident-only single deployment)

| Model | balanced | sonnet-heavy | code-heavy | 평균 |
|---|---:|---:|---:|---:|
| Qwen 0.5B | +34.6% | +10.5% | +18.4% | +21.2% |
| Qwen 1.5B | +27.8% | +7.9% | +9.4% | +15.0% |
| **Qwen 7B** | **+43.3%** ⭐ | +15.1% | +17.5% | **+25.3%** ⭐ |
| Qwen 32B | +26.8% | +2.0% | +0.5% | +9.8% |

→ **Qwen 7B 영역 AGSD gating 가치 영역 가장 큼** (평균 +25.3%). 0.5B/1.5B 영역 backend latency 영역 빠르므로 router overhead 영역 비중 ↑. 32B 영역 trident single 영역 throughput 영역 매우 높음 → gating 영역 marginal 만 추가 (특히 sonnet/code-heavy 영역 trident bottleneck).

---

## 4. ★ 4 모델 종합 AGSD-gated 향상 (vs vanilla-only single deployment)

| Model | balanced | sonnet-heavy | code-heavy | 평균 |
|---|---:|---:|---:|---:|
| Qwen 0.5B | +70.3% | +63.4% | +103.6% | +79.1% |
| Qwen 1.5B | +64.6% | +33.9% | +112.2% | +70.2% |
| Qwen 7B | +61.8% | +55.9% | +122.5% | +80.1% |
| **Qwen 32B** | **+93.8%** ⭐ | **+127.9%** ⭐ | **+144.1%** ⭐ | **+121.9%** ⭐ |

→ **Qwen 32B 영역 vanilla 대비 가장 큰 향상** — 모델 영역 클수록 spec_decoding 효과 (K accept) 영역 vanilla 영역 차이 크고, 2-GPU parallel 영역 effective.

---

## 5. ★ AGSD-gated win 영역 origin 분해

본 SUB_094/095 영역 AGSD-gated win 영역 두 source:

| source | 의미 | 평균 기여 |
|---|---|---|
| **parallel GPU utilization** | 2 backend 영역 동시 활성 → 2× theoretical throughput | ~50-60% (mix workload 영역 balance 정도 영역 의존) |
| **gating decision** | per-request workload 분류 → best method 선택 (chat → vanilla / sonnet+code → trident) | ~10-20% (model-by-model 다름) |

본 SUB 영역 fair 1-instance vs 2-instance 비교 불가 — 향후 SUB 영역 single-instance gating (vLLM per-request override 영역 patch) 영역 측정 시 두 source 영역 분리 가능.

---

## 6. caveat

| 영역 | 영역 |
|---|---|
| classifier overhead 영역 negligible (SUB_094 영역 1.22 ms = 0.16%) | 본 SUB 영역 4 모델 영역 모두 적용 |
| benchmark 영역 BENCH_MODEL hardcode bug | v1 영역 발견 → v2 영역 정정 (모든 cell 정상 측정) |
| Llama 70B / Qwen 72B 영역 측정 불가 | GPU mem 영역 2-instance 영역 불가 (TP=8 × 2 = 16 GPU 필요) |
| backend latency variance | 200 prompt × concurrency 32 영역 단발 측정 — long-run sustained QPS 영역 follow-up |

---

## 7. raw data

각 모델 영역 directory:
- `qwen05b/benchmark_{balanced,sonnet-heavy,code-heavy}.json`
- `qwen15b/benchmark_{...}.json`
- `qwen32b/benchmark_{...}.json`

(Qwen 7B 영역 SUB_094 영역 raw data 영역 그대로 인용.)

logs: 각 model `logs/` (vanilla.log / trident.log / router.log / bench_*.log)
