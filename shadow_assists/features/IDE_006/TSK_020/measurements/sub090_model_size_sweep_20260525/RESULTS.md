# SUB_090 — R/K model-size scaling sweep (Qwen 0.5B/1.5B/7B × code)

> **parent**: TSK_020 / R/K framework 영역 model-size 확장
> **measurement**: 2026-05-25 KST 11:18~11:26, 9 cell × code workload × 50p × 1024in × 512max
> **status**: ✅ 완료 — **R/K boundary 영역 7B 영역 아직 net negative, 7B~70B 사이 영역 boundary**

---

## 1. 측정 결과 (9 cell)

| model | TP | gmu | config | tps | wall (s) | vs vanilla |
|---|---:|---:|---|---:|---:|---:|
| Qwen 0.5B | 1 | 0.40 | vanilla | **11,220.2** | 2.3 | — |
| Qwen 0.5B | 1 | 0.40 | ngram cap=8 | 7,793.9 | 3.3 | **−30.5%** |
| Qwen 0.5B | 1 | 0.40 | suffix PIECEWISE | 5,376.2 | 4.8 | −52.1% |
| Qwen 1.5B | 1 | 0.40 | vanilla | **10,388.5** | 2.5 | — |
| Qwen 1.5B | 1 | 0.40 | ngram cap=8 | 5,855.0 | 4.4 | −43.6% |
| Qwen 1.5B | 1 | 0.40 | suffix PIECEWISE | 4,064.4 | 6.3 | −60.9% |
| **Qwen 7B** | 1 | 0.50 | **vanilla** | **5,556.2** ⭐ | 4.6 | — |
| **Qwen 7B** | 1 | 0.50 | **ngram cap=8** | 4,593.5 | 5.6 | **−17.3%** ⭐ (가장 적은 회귀) |
| Qwen 7B | 1 | 0.50 | suffix PIECEWISE | 3,515.5 | 7.3 | −36.7% |

(설정 참조: 모두 cudagraph_mode=PIECEWISE, fp8 kv, code workload)

## 2. ★ R/K boundary 분석

### 2.1 model-size × spec method matrix (vs vanilla, code workload)

| model | vanilla tps | ngram | suffix | (참조) large Llama-70B |
|---|---:|---:|---:|---:|
| Qwen 0.5B | 11,220 | −30.5% ✗ | −52.1% ✗ | — |
| Qwen 1.5B | 10,389 | −43.6% ✗ | −60.9% ✗ | — |
| **Qwen 7B** | **5,556** | **−17.3%** ✗ (boundary 근접) | −36.7% ✗ | — |
| (Llama 70B) | (7,710) | (+31.5% ✓) | (**+50.3% ✓**) | — |

→ **boundary 영역 7B 영역 70B 사이**. Qwen 7B 영역 ngram 영역 −17% 영역 가장 적은 회귀 — net positive 영역 가까움.

### 2.2 R/K framework 영역 model-size dependency 정량화

본 doc R/K 모델 (`spec_wall / vanilla_wall ≈ R / K`):

| model | T_target estimate | R estimate | K (ngram code, fixed γ=7) | R/K |
|---|---:|---:|---:|---:|
| Qwen 0.5B | ~0.1-0.2 ms | ~5-10 | ~1 (low α 영역 code) | **5-10** (net regression) |
| Qwen 1.5B | ~0.3-0.5 ms | ~3-7 | ~1 | **3-7** |
| Qwen 7B | ~1-2 ms | ~1.5-2.5 | ~1 | **1.5-2.5** (boundary 근접) |
| Llama 70B (TP=8) | ~70 ms | ~1.3 | ~1.1~5 (workload-dep) | **0.3-1.2** (net positive 가능) |

→ R 영역 model-size 영역 **inverse-scale** 영역 (small 영역 R 영역 큼, large 영역 R 영역 작음). K 영역 model-size 영역 weak dependent.
→ R/K boundary (= 1) 영역 7B 영역 70B 사이.

## 3. ★ 새 fact — PIECEWISE 영역 small model + ngram 영역 영역 효과

| model | config | SUB_079 (FULL_AND_PIECEWISE, default) | SUB_090 (PIECEWISE only) | 차이 |
|---|---|---:|---:|---:|
| Qwen 0.5B | ngram | -59.4% | **-30.5%** | **+28.9 pp** ⭐ |
| Qwen 1.5B | ngram | -62.0% | -43.6% | +18.4 pp |

→ **cudagraph PIECEWISE mode 영역 small model + ngram 영역 회귀 폭 영역 ~30% 영역 향상**. dynamic batch shape 영역 small model 영역 더 효율 (large model 영역 SUB_087 영역 비슷 영역 +14% 영역 향상).

## 4. ★ production 권장 갱신 (model-size 별)

| model size | 권장 spec method | 본 fact |
|---|---|---|
| **≤ 1.5B** | **vanilla 만** | 모든 spec method -30~-61% 회귀 |
| **1.5B → 7B** | **vanilla** (또는 ngram with PIECEWISE — small loss -17~-30% 영역 acceptable 영역 spec latency 영역 작은 환경) | 7B ngram -17% 영역 boundary 근접 |
| **7B → 32B 영역 추정** | **boundary** — 측정 필요 (Qwen 32B / Qwen 72B 영역 cache 영역 있음) | unknown |
| **≥ 70B** | **suffix PIECEWISE** (3 workload 모두 net positive, SUB_085 v2) | 본 fork 영역 best |

## 5. 후속 — boundary refinement

**SUB_090 영역 follow-up candidate**:
- Qwen 32B + TP=2/4 영역 code × 3 config — boundary 정확화 (7B → 32B → 70B 영역 어디서 net positive)
- 단 32B model load 시간 영역 더 길음 + memory 영역 TP 영역 필요

**즉시 production guidance**:
- production 영역 1B-7B 영역 model 영역 vanilla
- production 영역 70B 영역 suffix PIECEWISE (본 SUB_085 v2 best)

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| 9 result.json | `eval/results/20260525_111859_sub090_qwen{05b,15b,7b}_{vanilla,ngram,suffix}/result.json` |
| launcher | `/tmp/run_sub090_model_size_sweep.sh` |
| wrapper | `/tmp/run_sub078_wrap.py` |
| stdout | `/tmp/sub090.log` |
| summary | `/tmp/sub090_summary.tsv` |
