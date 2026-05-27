# 전체 성능 측정 결과 (SUB_093 + SUB_094 + SUB_095 + SUB_096)

> **scope**: 2026-05-25 ~ 2026-05-26 KST / **모든 모델 × 모든 workload × 모든 case + util**
> **총 측정 cell**: **129** (single-instance per-cell **93** + end-to-end 2-instance **36**)
> **raw CSV**: [`_ALL_MEASUREMENTS.csv`](_ALL_MEASUREMENTS.csv)

---

## A. Single-instance per-cell measurements (SUB_093 + SUB_096)

각 cell = 단일 vLLM instance × 1 config × 1 workload. util 영역 config-wide (해당 config 영역 모든 workload 평균).

### opt-125m (TP=1)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 8,008 | 7,360 | — | 2.1 / 2.2 / — | 0.3 / 0.3 / — |
| chat | 9,542 | 9,514 | — | 2.1 / 2.2 / — | 0.3 / 0.3 / — |
| code | 9,772 | 6,435 | — | 2.1 / 2.2 / — | 0.3 / 0.3 / — |

### Qwen2.5-0.5B (TP=1)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 4,234 | 4,317 | **10,921** ⭐ | 2.0 / 2.0 / 2.5 | 0.4 / 0.3 / 0.4 |
| chat | 6,576 | 5,257 | **7,694** ⭐ | 2.0 / 2.0 / 2.5 | 0.4 / 0.3 / 0.4 |
| code | 7,228 | 7,949 | **11,785** ⭐ | 2.0 / 2.0 / 2.5 | 0.4 / 0.3 / 0.4 |

### Qwen2.5-1.5B (TP=1)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 4,771 | 3,430 | **6,389** ⭐ | 2.0 / 1.9 / 2.3 | 0.6 / 0.7 / 0.6 |
| chat | 5,571 | 4,669 | **6,528** ⭐ | 2.0 / 1.9 / 2.3 | 0.6 / 0.7 / 0.6 |
| code | 6,786 | 5,158 | **11,295** ⭐ | 2.0 / 1.9 / 2.3 | 0.6 / 0.7 / 0.6 |

### starcoder2-3b (TP=1)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 5,758 | 5,007 | — | 3.5 / 3.8 / — | 0.9 / 0.6 / — |
| chat | 6,723 | 7,486 | — | 3.5 / 3.8 / — | 0.9 / 0.6 / — |
| code | 6,794 | 7,242 | — | 3.5 / 3.8 / — | 0.9 / 0.6 / — |

### Qwen2.5-7B (TP=1)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 5,584 | 3,200 | **5,714** ⭐ | 2.1 / 2.0 / 2.0 | 1.3 / 1.2 / 1.3 |
| chat | 4,557 | 2,777 | 4,253 | 2.1 / 2.0 / 2.0 | 1.3 / 1.2 / 1.3 |
| code | 6,196 | 5,902 | **8,071** ⭐ | 2.1 / 2.0 / 2.0 | 1.3 / 1.2 / 1.3 |

### Phi-3-medium-14B (TP=1)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 3,438 | 4,881 | **6,523** ⭐ | 2.0 / 1.8 / 1.8 | 11.3 / 9.4 / 9.0 |
| chat | 3,138 | 3,242 | **4,173** ⭐ | 2.0 / 1.8 / 1.8 | 11.3 / 9.4 / 9.0 |
| code | 3,340 | 5,140 | **5,085** ⭐ | 2.0 / 1.8 / 1.8 | 11.3 / 9.4 / 9.0 |
| mix-sh | 3,434 | 4,744 | **6,563** ⭐ | 2.0 / 1.8 / 1.8 | 11.3 / 9.4 / 9.0 |
| mix-bal | 3,374 | 4,762 | **7,312** ⭐ | 2.0 / 1.8 / 1.8 | 11.3 / 9.4 / 9.0 |
| mix-ch | 3,342 | 5,246 | **6,390** ⭐ | 2.0 / 1.8 / 1.8 | 11.3 / 9.4 / 9.0 |

### Qwen2.5-72B (TP=8)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 6,456 | 7,968 | **9,959** ⭐ | 5.8 / 5.4 / 5.3 | 93.2 / 82.0 / 74.7 |
| chat | 2,560 | 3,462 | **3,770** ⭐ | 5.8 / 5.4 / 5.3 | 93.2 / 82.0 / 74.7 |
| code | 6,227 | 5,766 | 5,941 | 5.8 / 5.4 / 5.3 | 93.2 / 82.0 / 74.7 |
| mix-sh | 5,832 | 6,586 | **8,795** ⭐ | 5.8 / 5.4 / 5.3 | 93.2 / 82.0 / 74.7 |
| mix-bal | 5,702 | 6,692 | **8,228** ⭐ | 5.8 / 5.4 / 5.3 | 93.2 / 82.0 / 74.7 |
| mix-ch | 6,231 | 5,989 | **8,491** ⭐ | 5.8 / 5.4 / 5.3 | 93.2 / 82.0 / 74.7 |

### Llama-3.3-70B (TP=8)

| workload | vanilla tps | ngram tps | **Trident core** tps | CPU% (vanilla / ngram / Trident) | GPU% (vanilla / ngram / Trident) |
|---|---:|---:|---:|---|---|
| sonnet | 7,679 | 10,759 | **11,677** ⭐ | 5.6 / 7.6 / 5.3 | 93.8 / 84.2 / 73.3 |
| chat | 2,267 | 3,244 | **3,830** ⭐ | 5.6 / 7.6 / 5.3 | 93.8 / 84.2 / 73.3 |
| code | 6,718 | 5,362 | **7,981** ⭐ | 5.6 / 7.6 / 5.3 | 93.8 / 84.2 / 73.3 |
| mix-sh | 6,326 | 7,933 | **10,298** ⭐ | 5.6 / 7.6 / 5.3 | 93.8 / 84.2 / 73.3 |
| mix-bal | 6,054 | 6,554 | **9,514** ⭐ | 5.6 / 7.6 / 5.3 | 93.8 / 84.2 / 73.3 |
| mix-ch | 6,495 | 5,491 | **9,457** ⭐ | 5.6 / 7.6 / 5.3 | 93.8 / 84.2 / 73.3 |

---

## B. End-to-end 2-instance + CPU router measurements (SUB_094 + SUB_095)

각 cell = **2 vLLM backend** (vanilla GPU N / Trident GPU M) + FastAPI router + concurrent client (200p × concurrency 32). util 영역 backend 영역 vLLM HTTP server 모드 영역 capture 없음.

### Qwen2.5-0.5B (TP=1 × 2 instances)

| Mix | vanilla-only tps | trident-only tps | **AGSD-gated** tps | vs vanilla | vs trident | backend split |
|---|---:|---:|---:|---:|---:|---|
| balanced | 3,672 | 4,644 | **6,252** ⭐ | **+70.2%** ⭐ | **+34.6%** ⭐ | gating routes per workload |
| sonnet-heavy | 4,196 | 6,207 | **6,858** ⭐ | **+63.4%** ⭐ | **+10.5%** ⭐ | gating routes per workload |
| code-heavy | 4,227 | 7,267 | **8,605** ⭐ | **+103.6%** ⭐ | **+18.4%** ⭐ | gating routes per workload |

### Qwen2.5-1.5B (TP=1 × 2 instances)

| Mix | vanilla-only tps | trident-only tps | **AGSD-gated** tps | vs vanilla | vs trident | backend split |
|---|---:|---:|---:|---:|---:|---|
| balanced | 3,512 | 4,524 | **5,783** ⭐ | **+64.6%** ⭐ | **+27.8%** ⭐ | gating routes per workload |
| sonnet-heavy | 4,068 | 5,050 | **5,449** ⭐ | **+34.0%** ⭐ | **+7.9%** ⭐ | gating routes per workload |
| code-heavy | 4,211 | 8,166 | **8,932** ⭐ | **+112.1%** ⭐ | **+9.4%** ⭐ | gating routes per workload |

### Qwen2.5-7B (TP=1 × 2 instances)

| Mix | vanilla-only tps | trident-only tps | **AGSD-gated** tps | vs vanilla | vs trident | backend split |
|---|---:|---:|---:|---:|---:|---|
| balanced | 3,753 | 4,238 | **6,073** ⭐ | **+61.8%** ⭐ | **+43.3%** ⭐ | gating routes per workload |
| sonnet-heavy | 3,865 | 5,234 | **6,025** ⭐ | **+55.9%** ⭐ | **+15.1%** ⭐ | gating routes per workload |
| code-heavy | 3,966 | 7,512 | **8,825** ⭐ | **+122.5%** ⭐ | **+17.5%** ⭐ | gating routes per workload |

### Qwen2.5-32B (TP=2 × 2 instances)

| Mix | vanilla-only tps | trident-only tps | **AGSD-gated** tps | vs vanilla | vs trident | backend split |
|---|---:|---:|---:|---:|---:|---|
| balanced | 1,779 | 2,721 | **3,449** ⭐ | **+93.8%** ⭐ | **+26.8%** ⭐ | gating routes per workload |
| sonnet-heavy | 1,792 | 4,004 | **4,084** ⭐ | **+127.9%** ⭐ | **+2.0%** ⭐ | gating routes per workload |
| code-heavy | 1,874 | 4,551 | **4,575** ⭐ | **+144.1%** ⭐ | **+0.5%** ⭐ | gating routes per workload |

---

## C. 종합 요약

### C.1 Trident core 효과 (single-instance, 6 workload 평균)

| Model | Trident vs vanilla 평균 | net positive 수 / 6 | 비고 |
|---|---:|:---:|---|
| opt-125m | (no agsd data) | — | TP=1 |
| Qwen2.5-0.5B | +79.3% | 3/3 | TP=1 |
| Qwen2.5-1.5B | +39.2% | 3/3 | TP=1 |
| starcoder2-3b | (no agsd data) | — | TP=1 |
| Qwen2.5-7B | +8.6% | 2/3 | TP=1 |
| Phi-3-medium-14B | +79.0% | 6/6 | TP=1 |
| Qwen2.5-72B | +38.1% | 5/6 | TP=8 |
| Llama-3.3-70B | +50.9% | 6/6 | TP=8 |

### C.2 AGSD end-to-end 효과 (2-instance, 3 mix 평균)

| Model | AGSD vs vanilla-only 평균 | AGSD vs trident-only 평균 | net positive |
|---|---:|---:|:---:|
| Qwen2.5-0.5B | +79.1% | +21.2% | 3/3 ⭐ |
| Qwen2.5-1.5B | +70.2% | +15.0% | 3/3 ⭐ |
| Qwen2.5-7B | +80.1% | +25.3% | 3/3 ⭐ |
| Qwen2.5-32B | +122.0% | +9.8% | 3/3 ⭐ |
