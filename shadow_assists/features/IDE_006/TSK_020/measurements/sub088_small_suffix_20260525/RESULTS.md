# SUB_088 — small model (Qwen 0.5B/1.5B) × suffix PIECEWISE 측정

> **parent**: TSK_020 / IDE_010 (SuffixDecoding) + IDE_014 (small model regression) 영역 확장
> **measurement**: 2026-05-25 KST 10:45~10:50, 6 cell × 50p × 1024in × 512max
> **status**: ✅ 완료 — **small model 영역 suffix decoding 도 net regression 확정**

---

## 1. 측정 결과

`/tmp/run_sub078_wrap.py` × method=suffix × cudagraph PIECEWISE × gmu=0.40 × TP=1:

| model | workload | tps | wall (s) | out_tok |
|---|---|---:|---:|---:|
| Qwen 0.5B | sonnet | 4,118.5 | 4.5 | 18,453 |
| Qwen 0.5B | chat | 4,460.7 | 5.1 | 22,549 |
| Qwen 0.5B | code | 5,374.7 | 4.8 | 25,600 |
| Qwen 1.5B | sonnet | 3,404.9 | 6.3 | 21,512 |
| Qwen 1.5B | chat | 4,392.5 | 5.0 | 21,789 |
| Qwen 1.5B | code | 4,154.2 | 6.2 | 25,600 |

> wall 짧음 (4-6s) → vLLM SpecDecoding metric interval (10s) 보다 짧아 acceptance metric 영역 미수집.

## 2. ★ small model fair comparison — vanilla / ngram / suffix

vs SUB_079 (vanilla + ngram, gmu=0.40, TP=1, eager mode 영역 verified):

| model | workload | vanilla (SUB_079) | ngram (SUB_079) vs vanilla | **suffix (SUB_088) vs vanilla** | suffix vs ngram |
|---|---|---:|---:|---:|---:|
| Qwen 0.5B | sonnet | 11,820.6 | 6,111.8 (−48.3%) | **4,118.5 (−65.2%)** | -32.6% (suffix 더 회귀) |
| Qwen 0.5B | chat | 13,675.5 | 4,745.9 (−65.3%) | **4,460.7 (−67.4%)** | -6.0% |
| **Qwen 0.5B** | **code** | **11,056.2** | **4,485.9 (−59.4%)** | **5,374.7 (−51.4%)** | **+19.8% (suffix 개선)** ⭐ |
| Qwen 1.5B | sonnet | 12,594.8 | 5,015.5 (−60.2%) | **3,404.9 (−73.0%)** | -32.1% |
| Qwen 1.5B | chat | 11,589.4 | 4,539.6 (−60.8%) | **4,392.5 (−62.1%)** | -3.2% |
| Qwen 1.5B | code | 11,015.5 | 4,195.1 (−62.0%) | **4,154.2 (−62.3%)** | -1.0% |

## 3. 핵심 결론 — small model 영역 suffix 도 vanilla 능가 못 함

- **모든 6 cell 영역 vanilla 대비 회귀** (-51.4 ~ -73.0%). **suffix decoding 영역 small model 영역 R≫K constraint 영역 못 극복**.
- code 영역만 suffix 영역 ngram 대비 +20% 개선 (suffix 의 prompt+generation pool advantage 영역 활용). 단 vanilla 영역 여전히 best.
- **production 권장 변경 없음**: small model (1B 이하) 영역 모든 spec method (ngram, suffix) 영역 OFF, vanilla 사용.

## 4. mechanism — small model 영역 R 가 큰 이유

본 doc R/K framework (small model 영역 PIECEWISE + suffix):
- T_target (small model 영역 forward) 매우 짧음 (~0.1-1 ms/step)
- spec step 영역 overhead R (kv cache copy + cudagraph capture/replay + sampler dispatch + suffix tree maintenance) 영역 forward time 영역 비교 영역 큰 비율
- K (suffix per-draft) 영역 large model 대비 못 더 큼 (acceptance 영역 model-dep 영역 model intent shift 영역 영향 영역 작음)
- → R / K > 1 영역 모든 workload 영역 net regression

본 SUB 영역 fact 영역 IDE_014 영역 "small model 영역 workload-universal regression" 명제 영역 **suffix decoding 까지 확장**.

## 5. 본 doc 갱신

- `analysis/workload_acceptance_analysis_20260524.md` §10.4 / §11.3 — small model 영역 suffix 도 안 됨 영역 cross-validation 추가
- `idea/IDE_010_suffix_decoding_measurement.md` §8 — small model 영역 suffix 영역 결과 추가
- `idea/IDE_014_issue_16258_repro.md` §7 — suffix 도 universal regression 확장

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| 6 result.json | `eval/results/20260525_104543_sub088_qwen{05b,15b}_{sonnet,chat,code}_suffix/` |
| launcher | `/tmp/run_sub088_small_suffix.sh` |
| wrapper | `/tmp/run_sub078_wrap.py` (재사용) |
| stdout | `/tmp/sub088.log` |
| summary | `/tmp/sub088_summary.tsv` |
