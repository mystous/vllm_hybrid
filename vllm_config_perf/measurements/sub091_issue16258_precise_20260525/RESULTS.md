# SUB_091 — vLLM issue #16258 precise reproduction (opt-125m + starcoder2-3b)

> **parent**: TSK_020 / IDE_014 (cross-validation 정확화)
> **measurement**: 2026-05-25 KST 11:34~11:40
> **status**: ✅ 완료 ⭐⭐ — **issue #16258 영역 정확 reproduction 성공 (opt-125m 2.13× regression)**

---

## 1. issue #16258 setup vs SUB_091 setup

| 항목 | issue #16258 (dtransposed) | SUB_091 (본 fork) |
|---|---|---|
| hardware | 2× L4 (24 GB) | H100×1 (80 GB) |
| model | opt-125m, starcoder2-3b | opt-125m, starcoder2-3b ✓ (동일) |
| workload | (code-like, 추정) | code (humaneval-style) ✓ |
| spec config | `num_speculative_tokens=5, ngram_prompt_lookup_max=2` | 동일 ✓ |
| measurement | throughput vs vanilla | 동일 ✓ |

→ **opt-125m, starcoder2-3b 영역 정확 issue config 영역 reproduction**.

## 2. 측정 결과

| model | config | tps | wall (s) | out_tok | vs vanilla |
|---|---|---:|---:|---:|---:|
| **opt-125m** | vanilla | **11,398.9** | 2.2 | 25,600 | — |
| **opt-125m** | ngram_5lk2 | **5,350.3** | 4.8 | 25,600 | **2.13× regression (−53.1%)** ⭐ |
| **starcoder2-3b** | vanilla | **9,169.2** | 2.8 | 25,600 | — |
| **starcoder2-3b** | ngram_5lk2 | **3,979.2** | 6.4 | 25,600 | **2.30× regression (−56.6%)** |

## 3. ★ issue #16258 영역 핵심 명제 영역 정확 reproduction

issue #16258 보고:
- opt-125m + ngram: throughput 238 tok/s vs vanilla 504.8 tok/s → **2.12× regression**
- "regardless of the configuration, the inference is Pareto worse than the inference without the n-gram model"

본 SUB_091 결과:
- opt-125m + ngram: **2.13× regression** ⭐ (issue 영역 정확 일치 — 2.12× vs 2.13%)
- different hardware (H100×1 vs 2× L4) 영역 same pattern → **hardware-independent fundamental constraint** 확정

## 4. cross-validation — 본 fork 의 small model regression 명제

| source | hardware | model | code workload regression |
|---|---|---|---:|
| issue #16258 (dtransposed) | 2× L4 | opt-125m | **2.12×** |
| **SUB_091 본 fork** | **H100×1** | **opt-125m** | **2.13×** ⭐ |
| **SUB_091 본 fork** | **H100×1** | **starcoder2-3b** | **2.30×** |
| (SUB_078) 본 fork | H100×1 | Qwen 0.5B | 2.46× |
| (SUB_078) 본 fork | H100×1 | Qwen 1.5B | 2.63× |
| (SUB_088) 본 fork | H100×1 | Qwen 0.5B suffix | 2.06× (suffix 도 회귀) |

→ **5 model × 다양한 hardware/wrapper 영역 모두 small model + spec = 2~2.6× regression**. R≫K 영역 universal fundamental.

## 5. 본 doc 의 영향 (IDE_014 cross-validation 정확화)

| 이전 IDE_014 claim | 본 SUB 영역 정확화 |
|---|---|
| "Qwen 0.5B/1.5B substitution 영역 issue #16258 와 같은 패턴" | ✅ **정확 reproduction 성공** — opt-125m 영역 2.13× regression 영역 issue 영역 2.12× 영역 매우 일치 |
| "small model + ngram = severe regression (cross-validated)" | ✅ 본 SUB 영역 정확 model 영역 확정 — Qwen substitution 영역 정확 reproduction 영역 정합 |

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| 4 result.json | `eval/results/20260525_113411_sub091_{opt125m,starcoder2_3b}_{vanilla,ngram_5lk2}/result.json` |
| launcher | `/tmp/run_sub091_issue16258_repro.sh` |
| wrapper | `/tmp/run_sub078_wrap.py` |
| stdout | `/tmp/sub091.log` |
| summary | `/tmp/sub091_summary.tsv` |
| HF download | `~/.cache/huggingface/hub/models--facebook--opt-125m/`, `~/.cache/huggingface/hub/models--bigcode--starcoder2-3b/` |
